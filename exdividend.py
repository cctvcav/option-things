import pandas as pd
import yfinance as yf
import datetime as dt
from pathlib import Path

date = dt.date.today()


date_str = f"{date.year}{date.month:02d}{date.day:02d}"
TICKER_FILE = "NEWsymbols.txt"
OUTPUT_FILE = f"{date_str}/dividends_last_year.csv"
DATE = Path(date_str)
DATE.mkdir(exist_ok=True)

INDEX_OUTPUT = f"{date_str}/index_yields.csv"

# index -> proxy ETF used to estimate the dividend yield (None = assume q = 0)
INDEX_PROXY = {
    "_SPX": "SPY", "_XSP": "SPY",
    "_NDX": "QQQ", "_XND": "QQQ",
    "_RUT": "IWM", "_MRUT": "IWM",
    "_RUI": "IWB",   # Russell 1000
    "_RLV": "IWD",   # Russell 1000 Value
    "_DJX": "DIA",
    "_OEX": "OEF", "_XEO": "OEF",   # S&P 100
    "_SPESG": "EFIV",               # S&P 500 ESG
    "_SPEQX": "SPY",                # unsure of this one, verify
    "_SIXB": "XLB",   # Materials
    "_SIXI": "XLI",   # Industrials
    "_SIXM": "XLF",   # Financials
    "_SIXRE": "XLRE", # Real estate
    "_SIXU": "XLU",   # Utilities
    "_SIXV": "XLV",   # Health care
    "_VIX": None,     # volatility index, no dividends
    "_CBTX": None, "_MBTX": None, "_NANOS": None,  # bitcoin indexes
}

# Read tickers: one per line, or comma/whitespace separated. Dedupe, preserve order.
# with open(TICKER_FILE) as f:
#     raw = f.read().replace(",", " ").split()
# tickers = list(dict.fromkeys(t.strip().upper() for t in raw if t.strip()))
symbols = [s.strip() for s in Path(TICKER_FILE).read_text().split() if s.strip()]
# symbols = [s.replace("_", "^") for s in symbols]
indexes = [s for s in symbols if s.startswith("_")]

rows = []
failed = []

def ttm_dividends(sym):
    """Dividends over the last 365 days as a Series (empty if none)."""
    divs = yf.Ticker(sym).dividends
    if divs.empty:
        return divs
    cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
    return divs[divs.index >= cutoff]

for i, sym in enumerate(symbols):
    try:
        sym = sym.replace(".", "-")
        divs = yf.Ticker(sym).dividends
        if divs.empty:
            continue  # non-payer
        cutoff = pd.Timestamp.now(tz=divs.index.tz) - pd.Timedelta(days=365)
        recent = divs[divs.index >= cutoff]
        for ex_date, amount in recent.items():
            rows.append({
                "ticker": sym,
                "ex_date": ex_date.date(),
                "dividend": float(amount),
            })
    except Exception as e:
        failed.append((sym, str(e)))
    if i % 100 == 0:
        print(f"{i}/{len(symbols)} done")

out = pd.DataFrame(rows, columns=["ticker", "ex_date", "dividend"])
out = out.sort_values(["ticker", "ex_date"]).reset_index(drop=True)
out.to_csv(OUTPUT_FILE, index=False)

# --- index dividend yields via proxy ETFs ---
idx_rows = []
for sym in indexes:
    proxy = INDEX_PROXY.get(sym)
    q = 0.0
    if proxy:
        try:
            spot = yf.Ticker(proxy).history(period="5d")["Close"].iloc[-1]
            q = float(ttm_dividends(proxy).sum() / spot)
        except Exception as e:
            failed.append((sym, str(e)))
            q = float("nan")
    elif sym not in INDEX_PROXY:
        print(f"No mapping for {sym}, defaulting to q = 0")
    idx_rows.append({"index": sym, "proxy": proxy, "div_yield": q})

pd.DataFrame(idx_rows).to_csv(INDEX_OUTPUT, index=False)

print(f"{len(symbols)} tickers, {out['ticker'].nunique()} paid dividends, {len(out)} rows written")
print(f"{len(indexes)} indexes, {len(idx_rows)} rows written")
if failed:
    print("Failed:", failed)