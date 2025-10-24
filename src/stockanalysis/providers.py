"""Optional third-party data connectors (Finnhub, FMP, NewsAPI).

All functions degrade gracefully to an empty result when credentials are
missing or a remote API call fails. This keeps the core agent usable without
extra configuration while making it easy to bolt on richer data feeds.
"""

from __future__ import annotations

import os
import textwrap
from typing import Dict, List, Optional

import requests


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")


def _safe_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> Optional[dict]:
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_finnhub_transcript_summary(ticker: str) -> Optional[str]:
    """Return a condensed summary from the most recent Finnhub transcript."""
    if not FINNHUB_API_KEY:
        return None

    url = "https://finnhub.io/api/v1/stock/transcripts"
    data = _safe_get(url, {"symbol": ticker, "token": FINNHUB_API_KEY})
    if not data or "data" not in data or not data["data"]:
        return None

    transcript = data["data"][0]
    highlights = []
    for section in transcript.get("content", [])[:5]:
        text = section.get("text")
        if text:
            highlights.append(text.strip())
        if len(highlights) >= 3:
            break

    if not highlights and transcript.get("summary"):
        highlights.append(transcript["summary"].strip())

    if not highlights:
        return None

    summary = " ".join(highlights)
    summary = textwrap.shorten(summary, width=320, placeholder="...")
    return summary


def get_finnhub_insider_trades(ticker: str, limit: int = 5) -> List[Dict[str, str]]:
    """Return recent insider transactions for the given ticker."""
    if not FINNHUB_API_KEY:
        return []

    url = "https://finnhub.io/api/v1/stock/insider-transactions"
    data = _safe_get(url, {"symbol": ticker, "token": FINNHUB_API_KEY})
    if not data or "data" not in data:
        return []

    trades = []
    for item in data["data"]:
        name = item.get("name") or "Unknown"
        share = item.get("share")
        change = item.get("change")
        transaction_price = item.get("transactionPrice")
        date = item.get("transactionDate")
        if not date or change is None:
            continue
        direction = "Buy" if change > 0 else "Sell"
        trades.append(
            {
                "date": date,
                "name": name,
                "direction": direction,
                "shares": f"{abs(change):,.0f}" if isinstance(change, (int, float)) else str(change),
                "price": f"${transaction_price:,.2f}" if isinstance(transaction_price, (int, float)) else "N/A",
            }
        )
        if len(trades) >= limit:
            break

    return trades


def get_fmp_forward_estimates(ticker: str, limit: int = 4) -> List[Dict[str, str]]:
    """Pull forward revenue/EPS estimates from Financial Modeling Prep."""
    if not FMP_API_KEY:
        return []

    url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{ticker.upper()}"
    data = _safe_get(url, {"apikey": FMP_API_KEY})
    if not data:
        return []

    rows: List[Dict[str, str]] = []
    for item in data[:limit]:
        period = item.get("period") or item.get("date")
        if not period:
            continue
        eps = item.get("eps")
        revenue = item.get("revenue")
        rows.append(
            {
                "period": period,
                "eps": f"{eps:.2f}" if isinstance(eps, (int, float)) else "N/A",
                "revenue": f"${revenue/1e9:.2f}B" if isinstance(revenue, (int, float)) else "N/A",
            }
        )

    return rows


def get_newsapi_articles(ticker: str, limit: int = 5) -> List[Dict[str, str]]:
    """Fetch global news via NewsAPI for additional coverage."""
    if not NEWSAPI_KEY:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} stock",
        "language": "en",
        "pageSize": limit,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    data = _safe_get(url, params)
    if not data or data.get("status") != "ok":
        return []

    articles: List[Dict[str, str]] = []
    for article in data.get("articles", []):
        title = article.get("title")
        link = article.get("url")
        if not title or not link:
            continue
        desc = article.get("description")
        published = article.get("publishedAt", "")[:10]
        source = article.get("source", {}).get("name", "NewsAPI")
        articles.append(
            {
                "title": title,
                "link": link,
                "summary": desc,
                "published": published,
                "publisher": source,
            }
        )
    return articles


def get_polygon_options_summary(ticker: str) -> Optional[Dict[str, float]]:
    """Return aggregate call/put open interest and volume via Polygon."""
    if not POLYGON_API_KEY:
        return None

    url = f"https://api.polygon.io/v3/reference/tickers/{ticker.upper()}?apiKey={POLYGON_API_KEY}"
    # Check if ticker exists to avoid unnecessary calls.
    reference = _safe_get(url)
    if reference is None:
        return None

    options_url = "https://api.polygon.io/v3/snapshot/options/{ticker}".format(ticker=ticker.upper())
    data = _safe_get(options_url, {"apiKey": POLYGON_API_KEY})
    if not data or "results" not in data:
        return None

    calls_volume = 0.0
    puts_volume = 0.0
    calls_oi = 0.0
    puts_oi = 0.0

    for option in data.get("results", []):
        details = option.get("details", {})
        if not details:
            continue
        option_type = details.get("contract_type")
        day = option.get("day", {})
        volume = day.get("volume", 0) or 0
        open_interest = day.get("open_interest", 0) or 0
        if option_type == "call":
            calls_volume += volume
            calls_oi += open_interest
        elif option_type == "put":
            puts_volume += volume
            puts_oi += open_interest

    if calls_volume == puts_volume == calls_oi == puts_oi == 0:
        return None

    return {
        "call_volume": calls_volume,
        "put_volume": puts_volume,
        "call_oi": calls_oi,
        "put_oi": puts_oi,
    }
