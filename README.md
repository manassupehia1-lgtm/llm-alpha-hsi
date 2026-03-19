# LLM-Driven Alpha Signal Generation in HSI Constituent Stocks

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Research-orange) ![Sharpe](https://img.shields.io/badge/OOS%20Sharpe-1.34-brightgreen)

**Solo Research Project · 2024**
Tools: Python · OpenAI API (GPT-4) · pandas · scikit-learn · XGBoost · backtrader · WRDS

---

## Overview

This project constructs a fully systematic, LLM-augmented alpha generation framework targeting the **Hang Seng Index (HSI) constituent stocks**. The core idea is to extract forward-looking sentiment signals from company filings and earnings call transcripts using **GPT-4**, then combine those signals with traditional price/volume features to train a supervised **long/short classifier**. The strategy is backtested over 2019 to 2023 on an out-of-sample basis using a strict walk-forward methodology.

Key results:
- **Sharpe Ratio: 1.34** (out-of-sample, 2019-2023)
- Statistically significant alpha over the HSI benchmark and a naive momentum baseline
- Submitted to **SSRN** and **arXiv (cs.CE)**

---

## Methodology

### 1. Data Collection

HSI constituent company annual reports, interim reports, and earnings call transcripts were scraped from HKEX filings and investor relations pages. Price and volume data were sourced via **WRDS** (Wharton Research Data Services).

### 2. Sentiment Scoring Pipeline (GPT-4)

Documents are chunked and passed to **GPT-4 via the OpenAI API** with a structured prompt that elicits analyst-style scores across five dimensions: revenue outlook, margin trajectory, management tone, competitive positioning, and macro risk sensitivity. Scores are aggregated into a composite document-level sentiment score in the range [-1, +1]. The pipeline is designed for real-time operation, processing new filings as they arrive.

### 3. Feature Engineering

**LLM sentiment features** include the raw score, score momentum (delta vs. prior period), score surprise vs. consensus, and cross-sectional z-score within the HSI universe. **Price/volume signals** include 1M/3M/6M price momentum, volume-weighted momentum, volatility-adjusted momentum, and 52-week high proximity. **Interaction features** include sentiment x momentum cross terms and earnings event indicators.

### 4. Signal Generation (XGBoost Classifier)

- Target: binary long/short signal derived from forward 1-month excess return quintiles (top 2 = long, bottom 2 = short, middle excluded)
- Model: XGBoostClassifier trained on rolling in-sample windows
- Hyperparameter tuning via Bayesian optimisation (Optuna)
- Walk-forward retraining: model retrained quarterly with strict no-look-ahead discipline

### 5. Backtesting (backtrader)

Equal-weighted long/short portfolio rebalanced monthly with 15 bps transaction cost per leg. Benchmarks: (i) HSI total return index; (ii) naive 6-month price momentum long/short.

---

## Results

| Metric | Strategy | HSI Benchmark | Momentum Baseline |
|---|---|---|---|
| Annualised Return | 18.7% | 3.2% | 9.4% |
| Annualised Volatility | 13.9% | 19.1% | 16.2% |
| **Sharpe Ratio** | **1.34** | 0.17 | 0.58 |
| Max Drawdown | -11.4% | -38.6% | -22.1% |
| Calmar Ratio | 1.64 | 0.08 | 0.43 |
| Hit Rate (monthly) | 58.3% | -- | 51.2% |

*Out-of-sample period: January 2019 to December 2023. Transaction costs included.*

---

## Repository Structure

```
llm-alpha-hsi/
├── data/
│   ├── raw/                         # Raw filings and transcripts (see WRDS)
│   └── processed/                   # Cleaned text and price data snapshots
├── src/
│   ├── scraper.py                   # HKEX filing scraper and transcript downloader
│   ├── sentiment_pipeline.py        # GPT-4 sentiment scoring pipeline (OpenAI API)
│   ├── feature_engineering.py       # Feature construction (sentiment + price/volume)
│   ├── model.py                     # XGBoost training and walk-forward validation
│   ├── backtest.py                  # backtrader strategy and performance analytics
│   └── utils.py                     # Shared utilities (logging, config, data loaders)
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis on filings corpus
│   ├── 02_sentiment_analysis.ipynb  # Sentiment score distributions and validation
│   ├── 03_feature_importance.ipynb  # SHAP-based feature attribution
│   └── 04_backtest_results.ipynb    # Full backtest tearsheet and benchmark comparison
├── configs/
│   └── config.yaml                  # Model hyperparameters and pipeline configuration
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/manassupehia1-lgtm/llm-alpha-hsi.git
cd llm-alpha-hsi
pip install -r requirements.txt
```

Set your API keys in `.env`:

```
OPENAI_API_KEY=your_openai_key
WRDS_USERNAME=your_wrds_username
```

Run the pipeline:

```bash
# 1. Score filings with GPT-4
python src/sentiment_pipeline.py --input data/raw/transcripts/ --output data/processed/scores.csv

# 2. Build features
python src/feature_engineering.py

# 3. Train model and run walk-forward validation
python src/model.py --config configs/config.yaml

# 4. Run backtest
python src/backtest.py --signals data/processed/signals.csv
```

---

## Citation

```bibtex
@misc{llm_alpha_hsi_2024,
  title  = {LLM-Driven Alpha Signal Generation in HSI Constituent Stocks},
  author = {Manassupehia1-lgtm},
  year   = {2024},
  note   = {SSRN / arXiv:cs.CE}
}
```

---

## References

- Brown, T. et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
- Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Ke, Z. T., Kelly, B., and Xiu, D. (2019). Predicting Returns with Text Data. *NBER Working Paper*.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

*This repository is for academic and research purposes only. Nothing here constitutes financial advice or a solicitation to trade.*
