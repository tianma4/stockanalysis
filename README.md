# StockAnalysis 🤖

StockAnalysis is an autonomous financial research agent that thinks, plans, and learns as it works. It performs analysis using task planning, self-reflection, and real-time market data. Think Claude Code, but built specifically for financial research.


<img width="979" height="651" alt="Screenshot 2025-10-14 at 6 12 35 PM" src="https://github.com/user-attachments/assets/5a2859d4-53cf-4638-998a-15cef3c98038" />

## Overview

StockAnalysis takes complex financial questions and turns them into clear, step-by-step research plans. It runs those tasks using live market data, checks its own work, and refines the results until it has a confident, data-backed answer.  

It’s not just another chatbot.  It’s an agent that plans ahead, verifies its progress, and keeps iterating until the job is done.

**Key Capabilities:**
- **Intelligent Task Planning**: Automatically decomposes complex queries into structured research steps
- **Autonomous Execution**: Selects and executes the right tools to gather financial data
- **Self-Validation**: Checks its own work and iterates until tasks are complete
- **Real-Time Financial Data**: Access to income statements, balance sheets, and cash flow statements
- **Safety Features**: Built-in loop detection and step limits to prevent runaway execution

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key
- Financial Datasets API key (get one at [financialdatasets.ai](https://financialdatasets.ai))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/virattt/stockanalysis.git
cd stockanalysis
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Set up your environment variables:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your-openai-api-key
# FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

### Usage

Run StockAnalysis in interactive mode:
```bash
uv run stockanalysis
```

Once the REPL is running you can:

- Ask natural language research questions (StockAnalysis will plan and execute tasks with the agent stack)
- Type a single stock ticker such as `AAPL` or `MSFT` to trigger an instant fundamentals brief.  
  The ticker analyzer fetches live financials, peer comps, valuation ratios, historical P/E trends, and a recent news digest without invoking the autonomous agent.
- Each brief now includes a market sentiment panel that blends headline tone, recent price momentum, and analyst activity so you can spot inflection points fast.
- Upcoming earnings, dividend dates, and sell-side consensus are surfaced automatically so you can prep for catalysts without leaving the terminal.
- A risk snapshot (annualized volatility, drawdown, Sharpe) turns StockAnalysis into a quick risk manager rather than just a quote screen.
- Optional API keys unleash richer context: Finnhub (call highlights & insider flow), FMP (forward estimates), NewsAPI (global headlines), and Polygon (call/put positioning).
- Manage a simple personal portfolio right from the prompt: `portfolio` shows live P/L, while `portfolio add/remove` keeps holdings up to date.

### Example Queries

Try asking StockAnalysis questions like:
- "What was Apple's revenue growth over the last 4 quarters?"
- "Compare Microsoft and Google's operating margins for 2023"
- "Analyze Tesla's cash flow trends over the past year"
- "What is Amazon's debt-to-equity ratio based on recent financials?"

StockAnalysis will automatically:
1. Break down your question into research tasks
2. Fetch the necessary financial data
3. Perform calculations and analysis
4. Provide a comprehensive, data-rich answer

## Architecture

StockAnalysis uses a multi-agent architecture with specialized components:

- **Planning Agent**: Analyzes queries and creates structured task lists
- **Action Agent**: Selects appropriate tools and executes research steps
- **Validation Agent**: Verifies task completion and data sufficiency
- **Answer Agent**: Synthesizes findings into comprehensive responses

## Project Structure

```
stockanalysis/
├── src/
│   ├── stockanalysis/
│   │   ├── agent.py      # Main agent orchestration logic
│   │   ├── model.py      # LLM interface
│   │   ├── tools.py      # Financial data tools
│   │   ├── prompts.py    # System prompts for each component
│   │   ├── schemas.py    # Pydantic models
│   │   ├── utils/        # Utility functions
│   │   └── cli.py        # CLI entry point
├── pyproject.toml
└── uv.lock
```

## Configuration

StockAnalysis supports configuration via the `Agent` class initialization:

```python
from stockanalysis.agent import Agent

agent = Agent(
    max_steps=20,              # Global safety limit
    max_steps_per_task=5       # Per-task iteration limit
)
```

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Expanding Data Sources

StockAnalysis ships with yfinance and Financial Datasets for core fundamentals. If you need deeper coverage, it’s straightforward to add providers such as:

- [Finnhub](https://finnhub.io) for earnings call transcripts, insider trades, and macro data (REST + WebSocket)
- [Financial Modeling Prep](https://financialmodelingprep.com) for forward estimates and sector aggregates
- [NewsAPI](https://newsapi.org) or [GDELT](https://www.gdeltproject.org) for richer news coverage beyond Yahoo headlines
- [X (Twitter) API](https://developer.twitter.com/) for real-time management commentary and market chatter
- [Polygon.io](https://polygon.io/) for options flow snapshots (call/put open interest & volume)

## Portfolio Tracker

StockAnalysis can keep tabs on a lightweight equity portfolio locally (stored in `portfolio.json`).

- `portfolio` — show holdings with live price, P/L %, and totals
- `portfolio add <ticker> <units> <cost_per_unit>` — create or update a position
- `portfolio add --total <ticker> <units> <total_cost>` — enter costs in aggregate instead
- `portfolio update <ticker> <units> <cost_per_unit>` — alias for `add`
- `portfolio remove <ticker>` — delete a holding

All prices come from yfinance; once you export your own holdings you’ll get instant P/L whenever you launch the REPL.

Create a new tool or extend the ticker analyzer to call these APIs once you add their credentials to `.env`.


## License

This project is licensed under the MIT License.
