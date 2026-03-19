"""
sentiment_pipeline.py
---------------------
GPT-4 sentiment scoring pipeline for HSI constituent company filings
and earnings call transcripts.

Usage:
    python src/sentiment_pipeline.py \
        --input data/raw/transcripts/ \
        --output data/processed/sentiment_scores.csv
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from utils import setup_logging, load_config, chunk_text

logger = logging.getLogger(__name__)


SENTIMENT_PROMPT = """You are an expert sell-side equity analyst specialising in Hong Kong and Chinese equities.

Analyse the following excerpt from a company filing or earnings call transcript and provide sentiment scores.

Score each dimension on a scale from -1.0 (very negative) to +1.0 (very positive):

1. revenue_outlook: Revenue and top-line growth trajectory
2. margin_trajectory: Gross/operating margin direction and management commentary
3. management_tone: Overall tone — confidence, caution, evasiveness
4. competitive_positioning: Market share, competitive moats, differentiation
5. macro_risk_sensitivity: Exposure to macro headwinds (rates, FX, regulation, geopolitics)

Respond ONLY with a valid JSON object in this exact format:
{
  "revenue_outlook": <float>,
  "margin_trajectory": <float>,
  "management_tone": <float>,
  "competitive_positioning": <float>,
  "macro_risk_sensitivity": <float>,
  "composite_score": <float>,
  "key_phrases": [<str>, <str>, <str>],
  "confidence": <float between 0 and 1>
}

Text excerpt:
---
{text}
---"""

DIMENSION_WEIGHTS = {
    "revenue_outlook": 0.30,
    "margin_trajectory": 0.25,
    "management_tone": 0.20,
    "competitive_positioning": 0.15,
    "macro_risk_sensitivity": 0.10,
}


class SentimentPipeline:
    """
    End-to-end pipeline for scoring company documents using GPT-4.
    
    Supports:
    - HKEX annual/interim reports (PDF text extraction)
    - Earnings call transcripts (.txt / .pdf)
    - Batch processing with rate limiting and retry logic
    """

    def __init__(self, config: dict):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = config.get("model", "gpt-4-turbo-preview")
        self.max_tokens = config.get("max_tokens", 1024)
        self.chunk_size = config.get("chunk_size", 3000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.max_chunks_per_doc = config.get("max_chunks_per_doc", 6)
        self.temperature = config.get("temperature", 0.1)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 5)
        self.rate_limit_delay = config.get("rate_limit_delay", 1.0)

    def score_text_chunk(self, text: str) -> Optional[dict]:
        """Score a single text chunk via GPT-4."""
        prompt = SENTIMENT_PROMPT.format(text=text[:4000])
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response.choices[0].message.content)
                time.sleep(self.rate_limit_delay)
                return result
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return None

    def score_document(self, text: str, ticker: str, doc_date: str, doc_type: str) -> dict:
        """
        Score an entire document by chunking and aggregating chunk scores.
        
        Returns a single aggregated sentiment record.
        """
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        chunks = chunks[: self.max_chunks_per_doc]
        logger.info(f"Scoring {ticker} {doc_type} {doc_date}: {len(chunks)} chunks")

        chunk_scores = []
        for chunk in chunks:
            score = self.score_text_chunk(chunk)
            if score:
                chunk_scores.append(score)

        if not chunk_scores:
            logger.error(f"No valid scores for {ticker} {doc_date}")
            return {}

        # Aggregate chunk scores (confidence-weighted average)
        agg = {}
        total_conf = sum(s.get("confidence", 1.0) for s in chunk_scores)
        
        for dim in DIMENSION_WEIGHTS:
            weighted = sum(
                s.get(dim, 0) * s.get("confidence", 1.0)
                for s in chunk_scores
            )
            agg[dim] = weighted / total_conf if total_conf > 0 else 0.0

        # Compute composite score
        composite = sum(
            agg[dim] * weight for dim, weight in DIMENSION_WEIGHTS.items()
        )
        
        # Clamp to [-1, 1]
        composite = max(-1.0, min(1.0, composite))

        return {
            "ticker": ticker,
            "date": doc_date,
            "doc_type": doc_type,
            "composite_score": composite,
            "n_chunks_scored": len(chunk_scores),
            **{f"dim_{k}": v for k, v in agg.items()},
        }

    def process_directory(self, input_dir: str, output_path: str) -> pd.DataFrame:
        """
        Process all documents in a directory structure.
        
        Expected directory layout:
            input_dir/
                {TICKER}/
                    {YYYY-MM-DD}_{doc_type}.txt
        """
        input_path = Path(input_dir)
        records = []

        all_files = list(input_path.rglob("*.txt"))
        logger.info(f"Found {len(all_files)} files to process")

        for fpath in tqdm(all_files, desc="Scoring documents"):
            try:
                ticker = fpath.parent.name.upper()
                stem_parts = fpath.stem.split("_", 1)
                doc_date = stem_parts[0]
                doc_type = stem_parts[1] if len(stem_parts) > 1 else "unknown"

                text = fpath.read_text(encoding="utf-8", errors="ignore")
                record = self.score_document(text, ticker, doc_date, doc_type)
                if record:
                    records.append(record)
            except Exception as e:
                logger.error(f"Error processing {fpath}: {e}")

        df = pd.DataFrame(records)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} sentiment records to {output_path}")
        return df


def main():
    parser = argparse.ArgumentParser(description="GPT-4 sentiment scoring pipeline")
    parser.add_argument("--input", required=True, help="Input directory with document files")
    parser.add_argument("--output", required=True, help="Output CSV path for sentiment scores")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)
    pipeline = SentimentPipeline(config.get("sentiment", {}))
    pipeline.process_directory(args.input, args.output)


if __name__ == "__main__":
    main()
