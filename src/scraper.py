"""
scraper.py
----------
Scraper for HKEX company filings (annual reports, interim reports)
and earnings call transcripts for HSI constituent stocks.

HKEX Filing API: https://www1.hkex.com.hk/hkexnews/

Usage:
    python src/scraper.py --tickers 0700.HK 0941.HK --output data/raw/
"""

import os
import re
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional
from datetime import datetime, date
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pdfplumber

from utils import setup_logging, get_hsi_constituents

logger = logging.getLogger(__name__)

HKEX_BASE_URL = "https://www1.hkex.com.hk"
HKEX_SEARCH_URL = "https://www1.hkex.com.hk/hkexnews/search/titlesearch.xhtml"

# Filing type codes on HKEX
FILING_TYPE_MAP = {
    "annual_report": "A",
    "interim_report": "I",
    "results_announcement": "RA",
}

REQUEST_DELAY = 1.5  # Polite crawling delay (seconds)


class HKEXFilingScraper:
    """
    Scrapes and processes regulatory filings from the HKEX disclosure portal.
    
    Extracts text from PDFs using pdfplumber and saves as .txt files
    organised by ticker and date.
    """

    def __init__(self, output_dir: str, delay: float = REQUEST_DELAY):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })

    def _get(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Make a GET request with retry logic."""
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=30, **kwargs)
                response.raise_for_status()
                time.sleep(self.delay)
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                if attempt < 2:
                    time.sleep(self.delay * (attempt + 2))
        return None

    def search_filings(
        self,
        ticker: str,
        filing_type: str,
        start_date: str,
        end_date: str,
    ) -> List[dict]:
        """
        Search HKEX for filings matching the given criteria.
        
        Returns list of dicts with keys: title, date, url, ticker
        """
        # Extract stock code (e.g. "0700" from "0700.HK")
        stock_code = ticker.split(".")[0].zfill(5)
        type_code = FILING_TYPE_MAP.get(filing_type, "A")
        
        params = {
            "sortdir": "0",
            "sortby": "0",
            "category": "1",
            "market": "HKEX",
            "stock_code": stock_code,
            "doc_type": type_code,
            "from_date": start_date.replace("-", ""),
            "to_date": end_date.replace("-", ""),
        }
        
        results = []
        response = self._get(HKEX_SEARCH_URL, params=params)
        if not response:
            return results
        
        soup = BeautifulSoup(response.content, "lxml")
        filing_rows = soup.find_all("tr", class_=re.compile("tr_odd|tr_even"))
        
        for row in filing_rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            
            date_text = cells[0].get_text(strip=True)
            title = cells[2].get_text(strip=True)
            link_tag = cells[2].find("a", href=True)
            
            if not link_tag:
                continue
            
            filing_url = urljoin(HKEX_BASE_URL, link_tag["href"])
            
            try:
                filing_date = datetime.strptime(date_text, "%d/%m/%Y").strftime("%Y-%m-%d")
            except ValueError:
                filing_date = date_text
            
            results.append({
                "ticker": ticker,
                "date": filing_date,
                "title": title,
                "url": filing_url,
                "filing_type": filing_type,
            })
        
        logger.info(f"Found {len(results)} {filing_type} filings for {ticker}")
        return results

    def download_pdf(self, url: str) -> Optional[bytes]:
        """Download a PDF file and return its raw bytes."""
        response = self._get(url)
        if response and "pdf" in response.headers.get("content-type", "").lower():
            return response.content
        return None

    def extract_pdf_text(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes using pdfplumber."""
        import io
        
        text_pages = []
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
        
        return "\n\n".join(text_pages)

    def scrape_ticker(
        self,
        ticker: str,
        filing_types: List[str],
        start_date: str,
        end_date: str,
    ) -> int:
        """
        Scrape and save all filings for a single ticker.
        Returns the number of successfully processed filings.
        """
        ticker_dir = self.output_dir / ticker.replace(".", "_")
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for filing_type in filing_types:
            filings = self.search_filings(ticker, filing_type, start_date, end_date)
            
            for filing in filings:
                output_file = ticker_dir / f"{filing['date']}_{filing_type}.txt"
                
                if output_file.exists():
                    logger.debug(f"Skipping existing: {output_file}")
                    continue
                
                pdf_bytes = self.download_pdf(filing["url"])
                if not pdf_bytes:
                    logger.warning(f"Could not download PDF: {filing['url']}")
                    continue
                
                text = self.extract_pdf_text(pdf_bytes)
                if len(text.strip()) < 100:
                    logger.warning(f"Very short text extracted for {ticker} {filing['date']}")
                    continue
                
                output_file.write_text(text, encoding="utf-8")
                logger.info(f"Saved: {output_file} ({len(text):,} chars)")
                count += 1
        
        return count


def main():
    parser = argparse.ArgumentParser(description="Scrape HKEX filings for HSI constituents")
    parser.add_argument("--tickers", nargs="+", help="Tickers to scrape (default: all HSI constituents)")
    parser.add_argument("--output", default="data/raw/", help="Output directory")
    parser.add_argument("--start", default="2017-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["annual_report", "interim_report"],
        choices=list(FILING_TYPE_MAP.keys()),
        help="Filing types to scrape",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    tickers = args.tickers or get_hsi_constituents(args.start)
    logger.info(f"Scraping {len(tickers)} tickers from {args.start} to {args.end}")
    
    scraper = HKEXFilingScraper(args.output)
    
    total = 0
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Processing {ticker}")
        try:
            n = scraper.scrape_ticker(ticker, args.types, args.start, args.end)
            total += n
        except Exception as e:
            logger.error(f"Failed to scrape {ticker}: {e}")
    
    logger.info(f"Scraping complete. Total filings saved: {total}")


if __name__ == "__main__":
    main()
