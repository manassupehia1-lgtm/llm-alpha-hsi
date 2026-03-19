"""
model.py
--------
XGBoost classifier for long/short signal generation with strict
walk-forward (expanding window) cross-validation.

No look-ahead bias: model retrained on Q1..Qn, evaluated on Q(n+1).

Usage:
    python src/model.py --config configs/config.yaml
"""

import logging
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from utils import setup_logging, load_config

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "mom_1m", "mom_3m", "mom_6m",
    "vw_momentum", "vol_adj_mom", "proximity_52w",
    "composite_score", "sentiment_delta", "sentiment_surprise", "sentiment_zscore",
    "sent_x_mom1m", "sent_x_mom3m", "sent_delta_x_mom1m",
]

LABEL_COL = "signal"
DATE_COL = "date"


def generate_walk_forward_splits(
    df: pd.DataFrame,
    min_train_quarters: int = 8,
    test_quarters: int = 1,
) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Expanding-window walk-forward splits by quarter.
    Training window grows by one quarter per fold; test is the next quarter.
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["quarter"] = df[DATE_COL].dt.to_period("Q")
    quarters = sorted(df["quarter"].unique())

    splits = []
    for i in range(min_train_quarters, len(quarters) - test_quarters + 1):
        train_q = quarters[:i]
        test_q = quarters[i: i + test_quarters]
        train_idx = df[df["quarter"].isin(train_q)].index
        test_idx = df[df["quarter"].isin(test_q)].index
        if len(train_idx) > 50 and len(test_idx) > 10:
            splits.append((train_idx, test_idx))

    logger.info(f"Generated {len(splits)} walk-forward splits")
    return splits


def objective(trial, X_train, y_train):
    """Optuna objective for XGBoost HPO via 3-fold CV."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    n = len(X_train)
    fold_size = n // 3
    cv_scores = []
    for fold in range(3):
        vs, ve = fold * fold_size, (fold + 1) * fold_size
        X_tr = np.concatenate([X_train[:vs], X_train[ve:]])
        y_tr = np.concatenate([y_train[:vs], y_train[ve:]])
        X_val, y_val = X_train[vs:ve], y_train[vs:ve]
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        try:
            cv_scores.append(roc_auc_score(y_val, preds))
        except ValueError:
            cv_scores.append(0.5)
    return float(np.mean(cv_scores))


def tune_hyperparameters(X_train, y_train, n_trials=50):
    """Run Optuna Bayesian HPO and return best params."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X_train, y_train), n_trials=n_trials)
    logger.info(f"Best HPO AUC: {study.best_value:.4f}")
    return study.best_params


def train_evaluate_fold(df, train_idx, test_idx, hpo_trials=50):
    """Train on one fold and return OOS metrics, signals, model, scaler."""
    X_tr = df.loc[train_idx, FEATURE_COLS].values
    y_tr = (df.loc[train_idx, LABEL_COL] == 1).astype(int).values
    X_te = df.loc[test_idx, FEATURE_COLS].values
    y_te = (df.loc[test_idx, LABEL_COL] == 1).astype(int).values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    best_params = tune_hyperparameters(X_tr, y_tr, n_trials=hpo_trials)
    best_params.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42})

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_tr, y_tr, verbose=False)

    proba = model.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    # Long/short signals from tercile thresholds
    th_hi, th_lo = np.percentile(proba, 67), np.percentile(proba, 33)
    raw_signals = np.where(proba >= th_hi, 1, np.where(proba <= th_lo, -1, 0))
    signal_series = pd.Series(raw_signals, index=test_idx, name="predicted_signal")

    metrics = {
        "accuracy": accuracy_score(y_te, preds),
        "roc_auc": roc_auc_score(y_te, proba),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall": recall_score(y_te, preds, zero_division=0),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
    return metrics, signal_series, model, scaler


def run_walk_forward(df, config):
    """Execute full walk-forward pipeline; return OOS signal DataFrame."""
    df = df[df[LABEL_COL].isin([1, -1])].reset_index(drop=True)
    splits = generate_walk_forward_splits(
        df,
        min_train_quarters=config.get("min_train_quarters", 8),
        test_quarters=config.get("test_quarters", 1),
    )
    all_signals, all_metrics = [], []
    for fold_idx, (tr, te) in enumerate(splits):
        logger.info(f"Fold {fold_idx+1}/{len(splits)} | train={len(tr)} test={len(te)}")
        try:
            metrics, signals, model, scaler = train_evaluate_fold(
                df, tr, te, hpo_trials=config.get("hpo_trials", 50)
            )
            metrics["fold"] = fold_idx + 1
            all_metrics.append(metrics)
            all_signals.append(signals)
            model_dir = Path(config.get("model_dir", "models/"))
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_dir / f"xgb_fold_{fold_idx+1}.pkl", "wb") as f:
                pickle.dump({"model": model, "scaler": scaler}, f)
        except Exception as e:
            logger.error(f"Fold {fold_idx+1} failed: {e}")

    oos = pd.concat(all_signals).rename("predicted_signal")
    df_out = df.join(oos, how="right")
    metrics_df = pd.DataFrame(all_metrics)
    logger.info("Walk-Forward Summary:\n" + metrics_df[["fold","roc_auc","accuracy"]].to_string())
    logger.info(f"Mean OOS AUC: {metrics_df['roc_auc'].mean():.4f}")
    return df_out, metrics_df


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost walk-forward model")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = pd.read_csv(cfg["data"]["features_path"], parse_dates=["date"])
    df_signals, metrics = run_walk_forward(df, cfg.get("model", {}))
    df_signals.to_csv(cfg["data"].get("signals_path", "data/processed/signals.csv"), index=False)
    metrics.to_csv("data/processed/wf_metrics.csv", index=False)
    logger.info("Walk-forward complete.")


if __name__ == "__main__":
    main()
