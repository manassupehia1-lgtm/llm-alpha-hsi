"""
utils.py
--------
Shared utilities: logging, config loading, text chunking,
and data helpers used throughout the pipeline.
"""

import logging
import sys
import re
from pathlib import Path
from typing import List, Optional
import yaml


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger with console and optional file handlers."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt, datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers,
    )
    for noisy in ("openai", "httpx", "urllib3", "backtrader"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(config_path: str) -> dict:
    """Load and return a YAML configuration file as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def clean_text(text: str) -> str:
    """Remove page numbers, headers, and collapse whitespace."""
    text = re.sub(r"-\s*\d+\s*-", "", text)
    text = re.sub(r"[Pp]age\s+\d+\s+of\s+\d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks at sentence boundaries.
    Ensures each chunk fits within GPT-4 context window per API call.
    """
    text = clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        slen = len(sentence)
        if current_length + slen > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            ot = " ".join(current_chunk)
            op = ot[max(0, len(ot) - overlap):]
            current_chunk = [op, sentence]
            current_length = len(op) + slen
        else:
            current_chunk.append(sentence)
            current_length += slen
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [c for c in chunks if len(c.strip()) > 50]


def get_hsi_constituents(date: str = None) -> List[str]:
    """
    Return HSI constituent tickers.
    In production, query WRDS or a stored constituents table keyed by date.
    """
    return [
        "0001.HK", "0002.HK", "0003.HK", "0005.HK", "0006.HK",
        "0011.HK", "0012.HK", "0016.HK", "0017.HK", "0019.HK",
        "0027.HK", "0066.HK", "0175.HK", "0267.HK", "0285.HK",
        "0288.HK", "0316.HK", "0386.HK", "0388.HK", "0669.HK",
        "0700.HK", "0762.HK", "0823.HK", "0857.HK", "0883.HK",
        "0939.HK", "0941.HK", "0960.HK", "0992.HK", "1038.HK",
        "1044.HK", "1093.HK", "1113.HK", "1177.HK", "1209.HK",
        "1211.HK", "1299.HK", "1398.HK", "1810.HK", "1876.HK",
        "1928.HK", "2007.HK", "2269.HK", "2318.HK", "2319.HK",
        "2382.HK", "2388.HK", "2628.HK", "3690.HK", "3988.HK",
        "6098.HK", "6862.HK", "9618.HK", "9888.HK", "9961.HK",
    ]


def load_prices(path: str):
    """Load price data CSV with standard forward-fill preprocessing."""
    import pandas as pd
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index().ffill().dropna(how="all")


def save_results(results: dict, output_path: str) -> None:
    """Persist a dict of results to a YAML file for human readability."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=True)
