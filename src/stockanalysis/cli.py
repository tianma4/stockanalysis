from dotenv import load_dotenv

# Load environment variables BEFORE importing any stockanalysis modules
load_dotenv()

import re
from typing import Optional

import yfinance as yf

from stockanalysis.agent import Agent
from stockanalysis.analyzers.ticker_analyzer import TickerAnalyzer
from stockanalysis.portfolio import Holding, load_portfolio, remove_holding, upsert_holding
from stockanalysis.utils.intro import print_intro
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory


TICKER_PATTERN = re.compile(r"^[A-Za-z]{1,5}(?:[\.\-][A-Za-z0-9]{1,4})?$")


def looks_like_ticker(query: str) -> bool:
    trimmed = query.strip()
    if " " in trimmed:
        return False
    return bool(TICKER_PATTERN.fullmatch(trimmed))


def _fetch_price(ticker: str) -> Optional[float]:
    try:
        yticker = yf.Ticker(ticker)
        info = yticker.info or {}
        for key in ("regularMarketPrice", "currentPrice", "previousClose"):
            val = info.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)

        fast_price = getattr(yticker, "fast_info", {}).get("lastPrice")
        if fast_price and fast_price > 0:
            return float(fast_price)

        hist = yticker.history(period="5d", auto_adjust=False)
        if not hist.empty:
            close = hist["Close"].dropna()
            if not close.empty:
                return float(close.iloc[-1])
    except Exception:
        return None
    return None


def _fmt_number(value: Optional[float], width: int = 12, precision: int = 2) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width},.{precision}f}"


def _fetch_fx_rate(base: str, quote: str) -> Optional[float]:
    """
    Fetch the most recent FX rate for base/quote via yfinance (e.g. USD/SGD).
    Returns None if the rate could not be retrieved.
    """
    ticker = f"{base}{quote}=X"
    try:
        rate_info = yf.Ticker(ticker)
        price = getattr(rate_info, "fast_info", {}).get("lastPrice")
        if price and price > 0:
            return float(price)

        info_price = rate_info.info.get("regularMarketPrice")
        if info_price and info_price > 0:
            return float(info_price)
    except Exception:
        return None
    return None


def _display_portfolio():
    holdings = load_portfolio()
    if not holdings:
        print("Portfolio is empty. Use 'portfolio add <ticker> <units> <cost>' to start tracking positions.")
        return

    usd_to_sgd = _fetch_fx_rate("USD", "SGD")

    header = (
        f"{'Ticker':<8}"
        f"{'Cost':>12}"
        f"{'Units':>10}"
        f"{'Total Cost':>14}"
        f"{'Current Price':>16}"
        f"{'Total Value':>14}"
        f"{'Total Value (SGD)':>20}"
        f"{'P/L':>14}"
        f"{'P/L (SGD)':>16}"
        f"{'P/L %':>10}"
    )
    print(header)
    print("-" * len(header))

    total_cost = 0.0
    total_value = 0.0
    total_value_sgd = 0.0
    total_pl_sgd = 0.0
    for holding in holdings:
        price = _fetch_price(holding.ticker)
        cost = holding.total_cost
        value = price * holding.units if price is not None else None
        pl = value - cost if value is not None else None
        pl_pct = (pl / cost) if pl is not None and cost else None
        value_sgd = value * usd_to_sgd if value is not None and usd_to_sgd is not None else None
        pl_sgd = pl * usd_to_sgd if pl is not None and usd_to_sgd is not None else None

        total_cost += cost
        if value is not None:
            total_value += value
            if value_sgd is not None:
                total_value_sgd += value_sgd
            if pl_sgd is not None:
                total_pl_sgd += pl_sgd

        print(
            f"{holding.ticker:<8}"
            f"{holding.cost_per_unit:>12,.2f}"
            f"{holding.units:>10,.2f}"
            f"{cost:>14,.2f}"
            f"{_fmt_number(price, 16)}"
            f"{_fmt_number(value, 14)}"
            f"{_fmt_number(value_sgd, 20)}"
            f"{_fmt_number(pl, 14)}"
            f"{_fmt_number(pl_sgd, 16)}"
            f"{_fmt_number((pl_pct * 100) if pl_pct is not None else None, 10)}"
        )

    print("-" * len(header))
    if total_cost:
        total_pl = total_value - total_cost
        total_pl_pct = total_pl / total_cost
        print(
            f"{'TOTAL':<8}"
            f"{'':>12}"
            f"{'':>10}"
            f"{total_cost:>14,.2f}"
            f"{'':>16}"
            f"{total_value:>14,.2f}"
            f"{_fmt_number(total_value_sgd if usd_to_sgd is not None else None, 20)}"
            f"{total_pl:>14,.2f}"
            f"{_fmt_number(total_pl_sgd if usd_to_sgd is not None else None, 16)}"
            f"{total_pl_pct*100:>10,.2f}"
        )
    else:
        print("No capital invested yet (zero total cost).")


def _handle_portfolio_command(raw_query: str) -> bool:
    tokens = raw_query.strip().split()
    if not tokens or tokens[0].lower() != "portfolio":
        return False

    if len(tokens) == 1:
        _display_portfolio()
        return True

    action = tokens[1].lower()
    if action in {"add", "update", "set"}:
        args = tokens[2:]
        mode = "total"
        if args and args[0] in {"--total", "--per-unit"}:
            mode = "per-unit" if args[0] == "--per-unit" else "total"
            args = args[1:]

        if len(args) < 3:
            print("Usage: portfolio add [--total|--per-unit] <ticker> <units> <cost>")
            return True

        ticker = args[0]
        try:
            units = float(args[1])
            cost_input = float(args[2])
        except ValueError:
            print("Units and cost must be numeric.")
            return True

        if units <= 0 or cost_input <= 0:
            print("Units and cost must be greater than zero.")
            return True

        if mode == "total":
            cost_per_unit = cost_input / units
            total_cost = cost_input
        else:  # per-unit
            cost_per_unit = cost_input
            total_cost = cost_per_unit * units

        holdings = upsert_holding(ticker, units, cost_per_unit)
        print(f"Saved holding for {ticker.upper()} ({units} units, total cost {total_cost:.2f}).")
        _display_portfolio()
        return True

    if action in {"remove", "delete"}:
        if len(tokens) < 3:
            print("Usage: portfolio remove <ticker>")
            return True
        ticker = tokens[2]
        holdings = remove_holding(ticker)
        print(f"Removed {ticker.upper()} from portfolio.")
        if holdings:
            _display_portfolio()
        else:
            print("Portfolio is now empty.")
        return True

    print("Portfolio commands: ")
    print("  portfolio                     -> show current holdings")
    print("  portfolio add <ticker> <units> <cost_per_unit>")
    print("  portfolio add --total <ticker> <units> <total_cost>")
    print("  portfolio remove <ticker>")
    print("  portfolio update <ticker> <units> <total_cost>")
    return True


def main():
    print_intro()
    agent = Agent()

    # Create a prompt session with history support
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            query = session.prompt(">> ")
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if query:
                if _handle_portfolio_command(query):
                    continue
                if looks_like_ticker(query):
                    try:
                        analyzer = TickerAnalyzer(query)
                        analyzer.generate_report()
                    except Exception as exc:
                        print(f"Failed to analyze {query}: {exc}")
                    continue
                agent.run(query)
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
