from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

PORTFOLIO_FILE = Path(__file__).resolve().parents[2] / "portfolio.json"


@dataclass
class Holding:
    ticker: str
    units: float
    cost_per_unit: float

    @property
    def total_cost(self) -> float:
        return self.units * self.cost_per_unit


def load_portfolio() -> List[Holding]:
    if not PORTFOLIO_FILE.exists():
        return []
    try:
        data = json.loads(PORTFOLIO_FILE.read_text())
    except json.JSONDecodeError:
        return []
    holdings = []
    for item in data.get("holdings", []):
        try:
            holdings.append(
                Holding(
                    ticker=item["ticker"].upper(),
                    units=float(item["units"]),
                    cost_per_unit=float(item["cost_per_unit"]),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return holdings


def save_portfolio(holdings: List[Holding]) -> None:
    data = {"holdings": [asdict(h) for h in holdings]}
    PORTFOLIO_FILE.write_text(json.dumps(data, indent=2))


def upsert_holding(ticker: str, units: float, cost_per_unit: float) -> List[Holding]:
    ticker = ticker.upper()
    holdings = load_portfolio()
    updated = False
    for idx, holding in enumerate(holdings):
        if holding.ticker == ticker:
            holdings[idx] = Holding(ticker=ticker, units=units, cost_per_unit=cost_per_unit)
            updated = True
            break
    if not updated:
        holdings.append(Holding(ticker=ticker, units=units, cost_per_unit=cost_per_unit))
    save_portfolio(holdings)
    return holdings


def remove_holding(ticker: str) -> List[Holding]:
    ticker = ticker.upper()
    holdings = [h for h in load_portfolio() if h.ticker != ticker]
    save_portfolio(holdings)
    return holdings

