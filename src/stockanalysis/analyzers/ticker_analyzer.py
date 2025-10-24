"""High-level utilities for generating a structured equity research brief."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
import html
import re
import textwrap
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stockanalysis.providers import (
    FINNHUB_API_KEY,
    FMP_API_KEY,
    NEWSAPI_KEY,
    POLYGON_API_KEY,
    get_finnhub_insider_trades,
    get_finnhub_transcript_summary,
    get_fmp_forward_estimates,
    get_newsapi_articles,
    get_polygon_options_summary,
)

from stockanalysis.utils.ui import Colors, UI


def _format_currency(value: Optional[float], currency: str = "USD") -> str:
    """Return human-friendly currency formatting."""
    if value is None:
        return "N/A"
    abs_value = abs(value)
    suffix = ""
    divider = 1.0
    for suffix, divider in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs_value >= divider:
            break
    else:
        suffix, divider = "", 1.0
    formatted = value / divider
    currency_symbol = "$" if currency.upper() in {"USD", "CAD", "AUD", "NZD"} else ""
    return f"{currency_symbol}{formatted:,.2f}{suffix}"


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:,.1f}%"


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return None


def _format_optional(value: Optional[float], fmt: str = "{:,.1f}") -> str:
    if value is None:
        return "N/A"
    try:
        return fmt.format(value)
    except Exception:
        return "N/A"


@dataclass
class ValuationContext:
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    price_to_book: Optional[float]
    peg_ratio: Optional[float]
    dividend_yield: Optional[float]
    price: Optional[float]
    market_cap: Optional[float]
    enterprise_value: Optional[float]
    currency: str


@dataclass
class FinancialSnapshot:
    revenue: Optional[float]
    revenue_growth: Optional[float]
    gross_margin: Optional[float]
    operating_margin: Optional[float]
    net_margin: Optional[float]
    free_cash_flow: Optional[float]
    cash: Optional[float]
    debt: Optional[float]


@dataclass
class PeerComparison:
    ticker: str
    name: str
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    revenue: Optional[float]
    market_cap: Optional[float]
    operating_margin: Optional[float]


@dataclass
class NewsItem:
    title: str
    publisher: str
    published: str
    link: str
    sentiment_label: str = "neutral"
    sentiment_score: Optional[float] = None
    summary: Optional[str] = None


@dataclass
class MarketSentiment:
    news_score: Optional[float]
    news_label: str
    price_1m_return: Optional[float]
    price_3m_return: Optional[float]
    analyst_consensus: Optional[str]
    analyst_recent: Optional[str]


@dataclass
class UpcomingEvents:
    earnings_date: Optional[str]
    earnings_consensus: Optional[str]
    revenue_consensus: Optional[str]
    dividend_date: Optional[str]
    ex_dividend: Optional[str]


@dataclass
class RiskMetrics:
    annual_volatility: Optional[float]
    max_drawdown: Optional[float]
    sharpe_ratio: Optional[float]


@dataclass
class PerformanceSnapshot:
    returns: Dict[str, Optional[float]]


def aggregate_news_summary(news_items: List[NewsItem]) -> str:
    positives = sum(1 for item in news_items if item.sentiment_label == "positive")
    negatives = sum(1 for item in news_items if item.sentiment_label == "negative")
    neutrals = len(news_items) - positives - negatives

    titles = [item.title for item in news_items[:5] if item.title]
    headline_snippet = "; ".join(titles)
    if len(headline_snippet) > 240:
        headline_snippet = headline_snippet[:237].rstrip() + "..."

    summary = (
        f"Coverage snapshot — Positive: {positives}, Neutral: {neutrals}, Negative: {negatives}. "
        f"Top themes: {headline_snippet if headline_snippet else 'No descriptive headlines available.'}"
    )
    return summary


class TickerAnalyzer:
    """Fetch structural data for a ticker and present a compact research brief."""

    def __init__(self, ticker: str):
        cleaned = ticker.strip().upper()
        if not cleaned:
            raise ValueError("Ticker symbol cannot be empty.")
        self.ticker = cleaned
        self._ui = UI()
        self._ticker = yf.Ticker(self.ticker)
        self._info = self._safe_call(self._ticker.get_info, default={})
        self._fast_info = self._safe_call(lambda: dict(self._ticker.fast_info), default={})
        self._currency = (
            self._fast_info.get("currency")
            or self._info.get("currency")
            or "USD"
        )
        self._shares_outstanding = (
            self._fast_info.get("sharesOutstanding")
            or self._info.get("sharesOutstanding")
        )
        self._sentiment = SentimentIntensityAnalyzer()

    def _safe_call(self, func, default=None):
        try:
            return func()
        except Exception:
            return default

    def _get_company_name(self) -> str:
        return (
            self._info.get("longName")
            or self._info.get("shortName")
            or self._info.get("displayName")
            or self.ticker
        )

    def _get_ceo_name(self) -> Optional[str]:
        officers = self._info.get("companyOfficers") or []
        for officer in officers:
            if not officer:
                continue
            title = (officer.get("title") or "").lower()
            name = officer.get("name")
            if not name or not title:
                continue
            if "chief executive officer" in title or title.startswith("ceo"):
                return name
        return None

    def _collect_financial_snapshot(self) -> FinancialSnapshot:
        financials = self._safe_call(lambda: self._ticker.financials)
        cashflow = self._safe_call(lambda: self._ticker.cashflow)
        balance_sheet = self._safe_call(lambda: self._ticker.balance_sheet)

        revenue, prev_revenue = self._extract_latest_metric(financials, "Total Revenue", periods=2)
        net_income, _ = self._extract_latest_metric(financials, "Net Income")
        gross_profit, _ = self._extract_latest_metric(financials, "Gross Profit")
        operating_income, _ = self._extract_latest_metric(financials, "Operating Income")

        operating_cash_flow, _ = self._extract_latest_metric(cashflow, "Total Cash From Operating Activities")
        capex, _ = self._extract_latest_metric(cashflow, "Capital Expenditures")
        free_cash_flow = None
        if operating_cash_flow is not None and capex is not None:
            free_cash_flow = operating_cash_flow + capex  # capex is usually negative

        cash, _ = self._extract_latest_metric(balance_sheet, "Cash And Cash Equivalents")
        total_debt, _ = self._extract_latest_metric(balance_sheet, "Total Debt")

        revenue_growth = _safe_div(revenue - prev_revenue, prev_revenue) if revenue and prev_revenue else None
        gross_margin = _safe_div(gross_profit, revenue)
        operating_margin = _safe_div(operating_income, revenue)
        net_margin = _safe_div(net_income, revenue)

        return FinancialSnapshot(
            revenue=revenue,
            revenue_growth=revenue_growth,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            free_cash_flow=free_cash_flow,
            cash=cash,
            debt=total_debt,
        )

    def _collect_valuation(self) -> ValuationContext:
        price = self._fast_info.get("lastPrice") or self._info.get("regularMarketPrice")
        market_cap = self._fast_info.get("marketCap") or self._info.get("marketCap")

        statistics = self._safe_call(
            lambda: self._ticker.get_key_stats(),
            default={}
        ) or {}

        financial_data = self._safe_call(
            lambda: self._ticker.get_financial_data(),
            default={}
        ) or {}

        enterprise_value = financial_data.get("enterpriseValue") or statistics.get("enterpriseValue")
        trailing_pe = self._info.get("trailingPE") or financial_data.get("trailingPE")
        forward_pe = self._info.get("forwardPE") or financial_data.get("forwardPE")
        price_to_book = self._info.get("priceToBook") or financial_data.get("priceToBook")
        peg_ratio = statistics.get("pegRatio") or self._info.get("pegRatio")
        dividend_yield = financial_data.get("dividendYield") or self._info.get("dividendYield")
        if dividend_yield and dividend_yield > 0.2:
            dividend_yield = dividend_yield / 100

        return ValuationContext(
            trailing_pe=trailing_pe,
            forward_pe=forward_pe,
            price_to_book=price_to_book,
            peg_ratio=peg_ratio,
            dividend_yield=dividend_yield,
            price=price,
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            currency=self._currency,
        )

    def _collect_historical_pe(self, years: int = 5) -> List[Tuple[int, Optional[float]]]:
        if not self._shares_outstanding:
            return []

        financials = self._safe_call(lambda: self._ticker.financials)
        if financials is None or financials.empty:
            return []

        if "Net Income" not in financials.index:
            return []

        net_income = financials.loc["Net Income"].dropna()
        if net_income.empty:
            return []

        price_history = self._safe_call(
            lambda: self._ticker.history(period=f"{years + 1}y", interval="1mo"),
            default=pd.DataFrame(),
        )
        if price_history.empty:
            return []

        price_history = price_history.copy()
        price_history.index = price_history.index.tz_localize(None)
        year_end_prices = price_history["Close"].resample("YE").last().dropna()

        output: List[Tuple[int, Optional[float]]] = []
        for date, income in net_income.items():
            year = date.year if isinstance(date, _dt.datetime) else int(str(date)[:4])
            if year < year_end_prices.index[0].year:
                continue
            try:
                year_price = year_end_prices[year_end_prices.index.year == year].iloc[-1]
            except IndexError:
                continue
            eps = _safe_div(income, self._shares_outstanding)
            pe = _safe_div(year_price, eps) if eps and eps > 0 else None
            output.append((year, pe))

        output = [entry for entry in output if entry[0] >= (_dt.datetime.now().year - years)]
        output.sort(key=lambda x: x[0], reverse=True)
        return output[:years]

    def _suggest_peers(self, limit: int = 6) -> List[str]:
        """Use Yahoo recommendations API to source similar tickers."""
        url = f"https://query2.finance.yahoo.com/v6/finance/recommendationsbysymbol/{self.ticker}"
        try:
            resp = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("finance", {}).get("result", [])
            if not results:
                return []
            symbols = [
                rec.get("symbol")
                for rec in results[0].get("recommendedSymbols", [])
                if rec.get("symbol")
            ]
            return symbols[:limit]
        except Exception:
            return []

    def _collect_peers(self, limit: int = 4) -> List[PeerComparison]:
        symbols = [
            symbol for symbol in self._suggest_peers(limit=limit * 2)
            if symbol and symbol.upper() != self.ticker
        ]
        peers: List[PeerComparison] = []
        for symbol in symbols[:limit]:
            peer_ticker = yf.Ticker(symbol)
            peer_info = self._safe_call(peer_ticker.get_info, default={})
            peer_fast = self._safe_call(lambda: dict(peer_ticker.fast_info), default={})
            peer_financials = self._safe_call(lambda: peer_ticker.financials, default=pd.DataFrame())

            revenue, _ = self._extract_latest_metric(peer_financials, "Total Revenue")
            operating_income, _ = self._extract_latest_metric(peer_financials, "Operating Income")
            operating_margin = _safe_div(operating_income, revenue)

            peers.append(
                PeerComparison(
                    ticker=symbol,
                    name=peer_info.get("shortName") or symbol,
                    trailing_pe=peer_info.get("trailingPE"),
                    forward_pe=peer_info.get("forwardPE"),
                    revenue=revenue,
                    market_cap=peer_fast.get("marketCap") or peer_info.get("marketCap"),
                    operating_margin=operating_margin,
                )
            )
        return peers

    def _collect_news(self, count: int = 5) -> List[NewsItem]:
        news_items: List[NewsItem] = []
        news_payload = self._safe_call(lambda: self._ticker.news, default=[]) or []

        for item in news_payload[:count]:
            title = item.get("title")
            link = item.get("link")
            if not title or not link:
                continue
            published = item.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published_dt = _dt.datetime.fromtimestamp(published)
                published_str = published_dt.strftime("%Y-%m-%d")
            else:
                published_str = str(published) if published else "N/A"

            sentiment_score, sentiment_label = self._score_sentiment(title)

            raw_summary = (
                item.get("summary")
                or item.get("description")
            )
            summary = self._clean_summary(raw_summary)

            news_items.append(
                NewsItem(
                    title=title,
                    publisher=item.get("publisher", "N/A"),
                    published=published_str,
                    link=link,
                    sentiment_label=sentiment_label,
                    sentiment_score=sentiment_score,
                    summary=summary,
                )
            )

        if not news_items:
            news_items.extend(self._fetch_google_news(self.ticker, limit=count))

        return news_items

    def _score_sentiment(self, text: str) -> tuple[Optional[float], str]:
        sentiment_score: Optional[float] = None
        sentiment_label = "neutral"
        try:
            score = self._sentiment.polarity_scores(text)["compound"]
            sentiment_score = score
            if score >= 0.1:
                sentiment_label = "positive"
            elif score <= -0.1:
                sentiment_label = "negative"
        except Exception:
            pass
        return sentiment_score, sentiment_label

    def _fetch_google_news(self, query: str, limit: int = 5) -> List[NewsItem]:
        url = (
            "https://news.google.com/rss/search?q="
            f"{quote_plus(query + ' stock')}"
            "&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception:
            return []

        items: List[NewsItem] = []
        for elem in root.findall(".//item"):
            title = elem.findtext("title")
            link = elem.findtext("link")
            if not title or not link:
                continue

            pub_date = elem.findtext("pubDate")
            published_str = "N/A"
            if pub_date:
                try:
                    parsed = parsedate_to_datetime(pub_date)
                    if parsed.tzinfo:
                        parsed = parsed.astimezone(_dt.timezone.utc)
                    published_str = parsed.strftime("%Y-%m-%d")
                except Exception:
                    published_str = pub_date

            source_elem = (
                elem.find("{http://www.w3.org/2005/Atom}source")
                or elem.find("source")
            )
            source = source_elem.text if source_elem is not None else "Google News"
            raw_summary = elem.findtext("description")
            summary = self._clean_summary(raw_summary)
            sentiment_score, sentiment_label = self._score_sentiment(title)

            items.append(
                NewsItem(
                    title=title,
                    publisher=source,
                    published=published_str,
                    link=link,
                    sentiment_label=sentiment_label,
                    sentiment_score=sentiment_score,
                    summary=summary,
                )
            )

            if len(items) >= limit:
                break

        return items

    def _clean_summary(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        # decode html entities and strip tags
        try:
            cleaned = html.unescape(text)
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                return None
            if len(cleaned) > 240:
                cleaned = cleaned[:237].rstrip() + "..."
            return cleaned
        except Exception:
            return None

    def _calculate_price_return(self, history: Optional[pd.DataFrame], periods: int) -> Optional[float]:
        if history is None or history.empty or "Close" not in history.columns:
            return None
        closes = history["Close"].dropna()
        if len(closes) <= periods:
            return None
        try:
            recent = closes.iloc[-1]
            prior = closes.iloc[-periods]
            return _safe_div(recent - prior, prior)
        except Exception:
            return None

    def _collect_market_sentiment(self, news_items: List[NewsItem]) -> MarketSentiment:
        news_scores: List[float] = []
        for item in news_items:
            if item.sentiment_score is not None:
                news_scores.append(item.sentiment_score)
        avg_score = mean(news_scores) if news_scores else None
        if avg_score is None:
            news_label = "neutral"
        elif avg_score >= 0.1:
            news_label = "positive"
        elif avg_score <= -0.1:
            news_label = "negative"
        else:
            news_label = "neutral"

        history = self._safe_call(lambda: self._ticker.history(period="6mo", interval="1d"), default=pd.DataFrame())
        price_1m = self._calculate_price_return(history, 21)
        price_3m = self._calculate_price_return(history, 63)

        analyst_consensus = None
        analyst_recent = None
        recommendations = self._safe_call(lambda: self._ticker.recommendations, default=pd.DataFrame())
        if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
            recs = recommendations.copy()
            recs.index = pd.to_datetime(recs.index, errors="coerce")
            recs = recs[recs.index.notna()]
            if not recs.empty:
                window = recs[recs.index >= (recs.index.max() - pd.Timedelta(days=90))]
                if window.empty:
                    window = recs.tail(20)
                if "To Grade" in window.columns and not window.empty:
                    counts = window["To Grade"].value_counts()
                    if not counts.empty:
                        analyst_consensus = counts.idxmax()
                latest_row = window.iloc[-1] if not window.empty else recs.iloc[-1]
                firm = latest_row.get("Firm")
                to_grade = latest_row.get("To Grade")
                action = latest_row.get("Action")
                detail = to_grade or action
                if action and to_grade:
                    detail = f"{to_grade} ({action})"
                if firm and detail:
                    analyst_recent = f"{firm}: {detail}"
                elif detail:
                    analyst_recent = detail

        return MarketSentiment(
            news_score=avg_score,
            news_label=news_label,
            price_1m_return=price_1m,
            price_3m_return=price_3m,
            analyst_consensus=analyst_consensus,
            analyst_recent=analyst_recent,
        )

    def _extract_latest_metric(
        self,
        frame: Optional[pd.DataFrame],
        label: str,
        periods: int = 1
    ) -> Tuple[Optional[float], Optional[float]]:
        if frame is None or frame.empty or label not in frame.index:
            return None, None
        series = frame.loc[label].dropna()
        if series.empty:
            return None, None
        latest_values = series.sort_index().iloc[-periods:]
        current = latest_values.iloc[-1]
        previous = latest_values.iloc[-2] if len(latest_values) > 1 else None
        return float(current), float(previous) if previous is not None else None

    def _collect_upcoming_events(self) -> UpcomingEvents:
        calendar = self._safe_call(lambda: self._ticker.calendar, default=None) or {}

        earnings_date = None
        raw_earnings_date = calendar.get("Earnings Date")
        if isinstance(raw_earnings_date, list) and raw_earnings_date:
            raw_earnings_date = raw_earnings_date[0]
        if isinstance(raw_earnings_date, (_dt.datetime, _dt.date)):
            earnings_date = raw_earnings_date.strftime("%Y-%m-%d")

        earnings_avg = calendar.get("Earnings Average")
        earnings_consensus = None
        if isinstance(earnings_avg, (int, float)):
            earnings_consensus = f"EPS {earnings_avg:.2f}"

        revenue_avg = calendar.get("Revenue Average")
        revenue_consensus = None
        if isinstance(revenue_avg, (int, float)):
            revenue_consensus = _format_currency(revenue_avg, self._currency)

        dividend_date = calendar.get("Dividend Date")
        if isinstance(dividend_date, (_dt.datetime, _dt.date)):
            dividend_date = dividend_date.strftime("%Y-%m-%d")
        else:
            dividend_date = None

        ex_dividend = calendar.get("Ex-Dividend Date")
        if isinstance(ex_dividend, (_dt.datetime, _dt.date)):
            ex_dividend = ex_dividend.strftime("%Y-%m-%d")
        else:
            ex_dividend = None

        return UpcomingEvents(
            earnings_date=earnings_date,
            earnings_consensus=earnings_consensus,
            revenue_consensus=revenue_consensus,
            dividend_date=dividend_date,
            ex_dividend=ex_dividend,
        )

    def _compute_risk_metrics(self) -> RiskMetrics:
        history = self._safe_call(
            lambda: self._ticker.history(period="1y", interval="1d"),
            default=pd.DataFrame(),
        )
        if history is None or history.empty or "Close" not in history.columns:
            return RiskMetrics(None, None, None)

        closes = history["Close"].dropna()
        returns = closes.pct_change().dropna()
        if returns.empty:
            return RiskMetrics(None, None, None)

        daily_std = returns.std()
        annual_vol = daily_std * (252 ** 0.5) if daily_std is not None else None

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = cumulative / rolling_max - 1
        max_drawdown = drawdown.min() if not drawdown.empty else None

        mean_daily = returns.mean()
        sharpe = None
        if daily_std and daily_std != 0:
            sharpe = (mean_daily / daily_std) * (252 ** 0.5)

        return RiskMetrics(
            annual_volatility=annual_vol,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
        )

    def _compute_performance_snapshot(self) -> PerformanceSnapshot:
        history = self._safe_call(
            lambda: self._ticker.history(period="5y", interval="1d"),
            default=pd.DataFrame(),
        )
        if history is None or history.empty or "Close" not in history.columns:
            return PerformanceSnapshot({})

        closes = history["Close"].dropna()
        if closes.empty:
            return PerformanceSnapshot({})

        latest_price = closes.iloc[-1]
        returns: Dict[str, Optional[float]] = {}

        def calc_return(days: int) -> Optional[float]:
            if len(closes) <= days:
                return None
            past_price = closes.iloc[-1 - days]
            return _safe_div(latest_price - past_price, past_price)

        # Map label to approximate trading-day counts.
        lookbacks = {
            "1D": 1,
            "1W": 5,
            "1M": 21,
            "1Q": 63,
            "1H": 126,
            "3Q": 189,
            "1Y": 252,
            "5Y": min(len(closes) - 1, 252 * 5),
        }

        for label, days in lookbacks.items():
            if days <= 0:
                returns[label] = None
                continue
            if label == "5Y" and days >= len(closes):
                returns[label] = None
            else:
                returns[label] = calc_return(days)

        return PerformanceSnapshot(returns)

    def _diagnose_valuation(
        self,
        valuation: ValuationContext,
        peers: Sequence[PeerComparison],
        historical_pe: Sequence[Tuple[int, Optional[float]]]
    ) -> Tuple[str, List[str]]:
        """Return qualitative valuation view and supporting evidence."""
        reasons: List[str] = []
        classification = "fairly valued"

        peer_pes = [p.trailing_pe for p in peers if p.trailing_pe]
        peer_avg = median(peer_pes) if peer_pes else None

        history_vals = [pe for _, pe in historical_pe if pe]
        history_avg = mean(history_vals) if history_vals else None

        trailing_pe = valuation.trailing_pe
        if trailing_pe and peer_avg:
            if trailing_pe >= peer_avg * 1.2:
                classification = "potentially overvalued"
                reasons.append(
                    f"Trailing P/E {trailing_pe:,.1f} is about {trailing_pe / peer_avg:,.1f}x peer average ({peer_avg:,.1f})."
                )
            elif trailing_pe <= peer_avg * 0.8:
                classification = "potentially undervalued"
                reasons.append(
                    f"Trailing P/E {trailing_pe:,.1f} is only {trailing_pe / peer_avg:,.1f}x peer average ({peer_avg:,.1f})."
                )

        if trailing_pe and history_avg:
            if trailing_pe >= history_avg * 1.3:
                classification = "potentially overvalued"
                reasons.append(
                    f"Trailing P/E {trailing_pe:,.1f} is {trailing_pe / history_avg:,.1f}x its 5y average ({history_avg:,.1f})."
                )
            elif trailing_pe <= history_avg * 0.7:
                classification = "potentially undervalued"
                reasons.append(
                    f"Trailing P/E {trailing_pe:,.1f} is {trailing_pe / history_avg:,.1f}x its 5y average ({history_avg:,.1f})."
                )

        if valuation.price_to_book and valuation.price_to_book > 5 and classification == "fairly valued":
            classification = "premium valuation"
            reasons.append(f"Price/Book {valuation.price_to_book:,.1f} suggests a premium vs assets.")

        if not reasons:
            reasons.append("Valuation multiples align with historical and peer ranges.")

        return classification, reasons

    def generate_report(self) -> str:
        """Return the formatted research brief and print it to the console."""
        company_name = self._get_company_name()

        with self._ui.progress(f"Collecting data for {self.ticker}...", "Data download complete"):
            financial_snapshot = self._collect_financial_snapshot()
            valuation = self._collect_valuation()
            historical_pe = self._collect_historical_pe()
            peers = self._collect_peers()
            news_items = self._collect_news()
            sentiment = self._collect_market_sentiment(news_items)
            events = self._collect_upcoming_events()
            risk_metrics = self._compute_risk_metrics()
            performance = self._compute_performance_snapshot()
            transcript_summary = get_finnhub_transcript_summary(self.ticker)
            insider_trades = get_finnhub_insider_trades(self.ticker)
            forward_estimates = get_fmp_forward_estimates(self.ticker)
            global_news = get_newsapi_articles(self.ticker)
            options_summary = get_polygon_options_summary(self.ticker)

        valuation_view, valuation_reasons = self._diagnose_valuation(valuation, peers, historical_pe)

        sections: List[Tuple[str, Iterable[str]]] = []

        # Snapshot
        snapshot_lines = [
            f"{company_name} ({self.ticker})",
            f"Price: {_format_currency(valuation.price, valuation.currency)}",
            f"Market Cap: {_format_currency(valuation.market_cap, valuation.currency)}",
        ]
        if self._info.get("sector"):
            snapshot_lines.append(f"Sector: {self._info.get('sector')} | Industry: {self._info.get('industry', 'N/A')}")
        if self._info.get("country"):
            snapshot_lines.append(f"Headquarters: {self._info.get('city', '')} {self._info.get('country')}".strip())
        ceo_name = self._get_ceo_name()
        if ceo_name:
            snapshot_lines.append(f"CEO: {ceo_name}")
        business_summary = self._info.get("longBusinessSummary")
        if business_summary:
            wrapped_summary = textwrap.fill(business_summary.strip(), width=100)
            snapshot_lines.append(f"Business overview:\n{wrapped_summary}")
        sections.append(("Snapshot", snapshot_lines))

        event_lines = []
        if events.earnings_date:
            event_lines.append(f"Next earnings: {events.earnings_date} ({events.earnings_consensus or 'consensus unavailable'})")
        if events.revenue_consensus:
            event_lines.append(f"Revenue consensus: {events.revenue_consensus}")
        if events.dividend_date:
            event_lines.append(f"Dividend payable: {events.dividend_date}")
        if events.ex_dividend:
            event_lines.append(f"Ex-dividend date: {events.ex_dividend}")
        if event_lines:
            sections.append(("Upcoming Events", event_lines))
        else:
            sections.append(("Upcoming Events", ["No upcoming events available from current data feeds."]))

        # Financials
        financial_lines = [
            f"Revenue (TTM/Annual): {_format_currency(financial_snapshot.revenue, valuation.currency)}",
            f"Revenue growth YoY: {_format_percent(financial_snapshot.revenue_growth)}",
            f"Gross margin: {_format_percent(financial_snapshot.gross_margin)}",
            f"Operating margin: {_format_percent(financial_snapshot.operating_margin)}",
            f"Net margin: {_format_percent(financial_snapshot.net_margin)}",
            f"Free cash flow: {_format_currency(financial_snapshot.free_cash_flow, valuation.currency)}",
            f"Cash: {_format_currency(financial_snapshot.cash, valuation.currency)}",
            f"Total debt: {_format_currency(financial_snapshot.debt, valuation.currency)}",
        ]
        sections.append(("Financial Highlights", financial_lines))

        valuation_lines = [
            f"Trailing P/E: {valuation.trailing_pe:,.1f}" if valuation.trailing_pe else "Trailing P/E: N/A",
            f"Forward P/E: {valuation.forward_pe:,.1f}" if valuation.forward_pe else "Forward P/E: N/A",
            f"Price/Book: {valuation.price_to_book:,.1f}" if valuation.price_to_book else "Price/Book: N/A",
            f"PEG Ratio: {valuation.peg_ratio:,.2f}" if valuation.peg_ratio else "PEG Ratio: N/A",
            f"Dividend yield: {_format_percent(valuation.dividend_yield)}",
            f"Enterprise Value: {_format_currency(valuation.enterprise_value, valuation.currency)}",
            f"Valuation view: {valuation_view}",
        ]
        valuation_lines.extend(valuation_reasons)
        sections.append(("Valuation", valuation_lines))

        sentiment_lines = [
            f"Headline tone: {sentiment.news_label.title()}" + (f" (avg score {sentiment.news_score:+.2f})" if sentiment.news_score is not None else ""),
            f"1M price return: {_format_percent(sentiment.price_1m_return)}",
            f"3M price return: {_format_percent(sentiment.price_3m_return)}",
        ]
        if sentiment.analyst_consensus:
            sentiment_lines.append(f"Analyst consensus (90d): {sentiment.analyst_consensus}")
        if sentiment.analyst_recent:
            sentiment_lines.append(f"Latest analyst note: {sentiment.analyst_recent}")
        sections.append(("Market Sentiment", sentiment_lines))

        sharpe_display = _format_optional(risk_metrics.sharpe_ratio, "{:.2f}")
        risk_lines = [
            f"Annual volatility: {_format_percent(risk_metrics.annual_volatility)}",
            f"Max drawdown (1Y): {_format_percent(risk_metrics.max_drawdown)}",
            f"Sharpe ratio (1Y): {sharpe_display}",
        ]
        sections.append(("Risk Snapshot", risk_lines))

        perf_lines = [
            f"1D: {_format_percent(performance.returns.get('1D'))}",
            f"1W: {_format_percent(performance.returns.get('1W'))}",
            f"1M: {_format_percent(performance.returns.get('1M'))}",
            f"1Q: {_format_percent(performance.returns.get('1Q'))}",
            f"1H: {_format_percent(performance.returns.get('1H'))}",
            f"3Q: {_format_percent(performance.returns.get('3Q'))}",
            f"1Y: {_format_percent(performance.returns.get('1Y'))}",
            f"5Y: {_format_percent(performance.returns.get('5Y'))}",
        ]
        sections.append(("Performance Ladder", perf_lines))

        if historical_pe:
            hist_lines = [
                f"{year}: {pe:,.1f}x" if pe else f"{year}: N/A"
                for year, pe in historical_pe
            ]
            sections.append(("Historical P/E (Year-End)", hist_lines))

        if peers:
            peer_lines = [
                f"{peer.ticker}: PE {_format_optional(peer.trailing_pe)} | Rev {_format_currency(peer.revenue, valuation.currency)} | Oper margin {_format_percent(peer.operating_margin)}"
                for peer in peers
            ]
            sections.append(("Peer Comparison", peer_lines))

        if transcript_summary:
            sections.append(("Earnings Call Takeaways", [transcript_summary]))
        elif FINNHUB_API_KEY:
            sections.append(("Earnings Call Takeaways", ["No recent transcript available from Finnhub."]))

        if insider_trades:
            insider_lines = [
                f"{trade['date']}: {trade['name']} {trade['direction']} {trade['shares']} @ {trade['price']}"
                for trade in insider_trades
            ]
            sections.append(("Recent Insider Activity", insider_lines))
        elif FINNHUB_API_KEY:
            sections.append(("Recent Insider Activity", ["No insider transactions reported in the latest Finnhub feed."]))

        if forward_estimates:
            estimate_lines = [
                f"{row['period']}: EPS {row['eps']} | Revenue {row['revenue']}"
                for row in forward_estimates
            ]
            sections.append(("Forward Estimates (FMP)", estimate_lines))
        elif FMP_API_KEY:
            sections.append(("Forward Estimates (FMP)", ["No forward estimates returned by FMP for this ticker."]))

        if options_summary:
            call_vol = options_summary.get("call_volume")
            put_vol = options_summary.get("put_volume")
            call_oi = options_summary.get("call_oi")
            put_oi = options_summary.get("put_oi")
            volume_total = (call_vol or 0) + (put_vol or 0)
            oi_total = (call_oi or 0) + (put_oi or 0)
            vol_ratio = (call_vol / volume_total) if volume_total else None
            oi_ratio = (call_oi / oi_total) if oi_total else None

            options_lines = [
                f"Call volume: {call_vol:,.0f} | Put volume: {put_vol:,.0f}",
                f"Call OI: {call_oi:,.0f} | Put OI: {put_oi:,.0f}",
                f"Volume call %: {vol_ratio:.1%}" if vol_ratio is not None else "Volume call %: N/A",
                f"OI call %: {oi_ratio:.1%}" if oi_ratio is not None else "OI call %: N/A",
            ]
            sections.append(("Options Flow", options_lines))
        elif POLYGON_API_KEY:
            sections.append(("Options Flow", ["No options snapshot returned by Polygon (possibly illiquid). "]))

        if news_items:
            highlight_lines = []
            seen_titles = set()
            for item in news_items:
                normalized_title = item.title.strip().lower()
                if normalized_title in seen_titles:
                    continue
                seen_titles.add(normalized_title)
                tone = item.sentiment_label.title()
                if item.sentiment_score is not None:
                    tone += f" ({item.sentiment_score:+.2f})"
                highlight_lines.append(
                    f"{item.published} — {item.title} [{tone}]"
                )
                if item.summary:
                    highlight_lines.append(f"    Summary: {item.summary}")
                if len(seen_titles) >= 3:
                    break
        else:
            highlight_lines = [
                "No Yahoo Finance headlines available for highlights."
            ]
        sections.append(("Latest News Highlights", highlight_lines))

        if news_items:
            summary_lines = []
            agg_summary = aggregate_news_summary(news_items)
            summary_lines.append(agg_summary)
            sections.append(("Recent News", summary_lines))
        else:
            sections.append(("Recent News", [
                "No Yahoo Finance headlines returned. Consider augmenting with NewsAPI or GDELT for broader coverage."
            ]))

        if global_news:
            global_lines = []
            for article in global_news:
                tone_score, tone_label = self._score_sentiment(article.get("title", ""))
                tone = tone_label.title()
                if tone_score is not None:
                    tone += f" ({tone_score:+.2f})"
                line = f"{article.get('published', 'N/A')} — {article.get('publisher', 'NewsAPI')}: {article['title']} [{tone}] ({article['link']})"
                global_lines.append(line)
                summary = self._clean_summary(article.get("summary"))
                if summary:
                    global_lines.append(f"    Summary: {summary}")
            sections.append(("Global News (NewsAPI)", global_lines))
        elif NEWSAPI_KEY:
            sections.append(("Global News (NewsAPI)", ["No NewsAPI articles retrieved (rate limited or empty response)."]))

        # Develop a one-liner summary that captures tone mix across all news feeds.
        def overall_tone(items: List[NewsItem]) -> str:
            pos = sum(1 for item in items if item.sentiment_label == "positive")
            neg = sum(1 for item in items if item.sentiment_label == "negative")
            neu = len(items) - pos - neg
            return f"Tone mix — Positive: {pos}, Neutral: {neu}, Negative: {neg}."

        all_news_items = news_items.copy()
        if global_news:
            for article in global_news:
                score, label = self._score_sentiment(article.get("title", ""))
                all_news_items.append(
                    NewsItem(
                        title=article.get("title", ""),
                        publisher=article.get("publisher", "NewsAPI"),
                        published=article.get("published", "N/A"),
                        link=article.get("link", ""),
                        sentiment_label=label,
                        sentiment_score=score,
                        summary=self._clean_summary(article.get("summary"))
                    )
                )

        if all_news_items:
            news_brief = aggregate_news_summary(all_news_items)
            sections.append(("News Overview", [overall_tone(all_news_items), news_brief]))

        report_lines: List[str] = []
        for title, rows in sections:
            report_lines.append(f"{title}:")
            for row in rows:
                report_lines.append(f"  - {row}")
            print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
            for row in rows:
                print(f"{Colors.BLUE}•{Colors.ENDC} {row}")

        compiled = "\n".join(report_lines)
        return compiled
