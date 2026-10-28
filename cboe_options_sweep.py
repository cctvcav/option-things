"""
Sweep Cboe delayed-quote option chains (cdn.cboe.com) for a list of underlyings.

Outputs:
  options_snapshot.csv      one row per option contract (IV, greeks, bid/ask, volume, OI)
  underlyings_snapshot.csv  one row per underlying (price, volume, aggregated option OI/volume)

Input:
  symbols.txt               whitespace/newline-separated underlyings (index symbols use a
                            leading underscore, e.g. _SPX)

Notes:
  - Unofficial, undocumented endpoint. Check Cboe's terms before running regularly.
  - JSON field names are from memory: run one symbol and inspect the raw JSON in
    cboe_cache/ to confirm they match before a full sweep.
  - Responses are cached to disk, so reruns cost no requests. Delete cboe_cache/ to
    start a fresh snapshot.
"""

import json
import random
import re
import time
import datetime
from pathlib import Path
import shutil

import pandas as pd
import requests

date = datetime.date.today()
date_str = f"{date.year}{date.month:02d}{date.day:02d}"
start_date_str = str(datetime.datetime.now())

BASE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
DATE = Path(date_str)
CACHE = DATE / Path("cboe_cache")
# MISC = DATE / Path("misc")
DATE.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
# MISC.mkdir(exist_ok=True)

# Put a real contact address here.
HEADERS = {"User-Agent": "options-research"}

MIN_DELAY, MAX_DELAY = 0.5, 1.5  # ~1 request/sec with jitter
LONG_PAUSE_EVERY = 300           # occasional longer pause (symbols)
LONG_PAUSE_RANGE = (5, 10)       # seconds

OCC_RE = re.compile(r"^(?P<root>.+?)(?P<exp>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

session = requests.Session()
session.headers.update(HEADERS)


def fetch_chain(sym: str, max_retries: int = 4):
    """Return parsed JSON for a symbol, using the disk cache when possible."""
    path = CACHE / f"{sym}.json"
    if path.exists():
        return json.loads(path.read_text())

    backoff = 300  # 5 minutes after the first block, doubling each retry
    for _ in range(max_retries):
        try:
            r = session.get(BASE.format(sym=sym), timeout=20)
        except requests.RequestException as e:
            print(f"{sym}: network error {e}")
            time.sleep(30)
            continue

        if r.status_code == 200:
            path.write_text(r.text)
            return r.json()
        if r.status_code == 404:  # no listed options / unknown symbol
            path.write_text(f"{sym}: HTTP {r.status_code}")
            return {}
        if r.status_code == 429 or r.status_code >= 500:
            print(f"{sym}: HTTP {r.status_code}, sleeping {backoff}s")
            time.sleep(backoff)
            backoff *= 2
            continue
        if r.status_code == 403:
            print(f"{sym}: HTTP {r.status_code}")
            if sym[0] != "_":
                print(f"Checking if {sym} is an index.")
                new_sym = f"_{sym}"
                new_chain = fetch_chain(new_sym, 1)
                if new_chain:
                    print(f"{sym} is an index.")
                    with open(f"{date_str}/eq_to_idx.txt", "a") as f:
                        f.write(f"{sym}\n{new_sym}\n")
                    return new_chain
                else:
                    print(f"{sym} is probably not an index")
                    with open(f"{date_str}/unknowns.txt", "a") as f:
                        f.write(f"{sym}\n")
                    return {}
            print(f"{sym} unable to be found")
            return {}
            
        print(f"{sym}: unexpected HTTP {r.status_code}")
        return None

    raise RuntimeError(f"Giving up on {sym}; stop for the day and slow down.")


def parse_chain(sym: str, payload: dict) -> list[dict]:
    rows = []
    for o in (payload.get("data") or {}).get("options", []):
        m = OCC_RE.match(o.get("option", ""))
        if not m:
            continue
        rows.append({
            "underlying": sym,
            "contract": o["option"],
            "expiry": pd.to_datetime(m["exp"], format="%y%m%d"),
            "type": m["cp"],
            "strike": int(m["strike"]) / 1000,
            "bid": o.get("bid"),
            "ask": o.get("ask"),
            "volume": o.get("volume"),
            "open_interest": o.get("open_interest"),
            "iv": o.get("iv"),
            "delta": o.get("delta"),
            "gamma": o.get("gamma"),
            "theta": o.get("theta"),
            "vega": o.get("vega"),
            "rho": o.get("rho"),
        })
    return rows


def parse_underlying(sym: str, payload: dict, option_rows: list[dict]) -> dict | None:
    d = payload.get("data") or {}
    if not d:
        return None

    opts = pd.DataFrame(option_rows)
    has_opts = not opts.empty
    if has_opts:
        opts["open_interest"] = pd.to_numeric(opts["open_interest"], errors="coerce")
        opts["volume"] = pd.to_numeric(opts["volume"], errors="coerce")

    return {
        "underlying": sym,
        "price": d.get("current_price"),
        "price_change": d.get("price_change"),
        "bid": d.get("bid"),
        "ask": d.get("ask"),
        "stock_volume": d.get("volume"),
        "iv30": d.get("iv30"),
        "last_trade_time": d.get("last_trade_time"),
        # Aggregated across all parsed option contracts (before liquidity filter)
        "num_contracts": len(opts),
        "total_option_oi": opts["open_interest"].sum() if has_opts else 0,
        "total_option_volume": opts["volume"].sum() if has_opts else 0,
        "call_oi": opts.loc[opts["type"] == "C", "open_interest"].sum() if has_opts else 0,
        "put_oi": opts.loc[opts["type"] == "P", "open_interest"].sum() if has_opts else 0,
    }


def sweep(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    option_rows_all, underlying_rows = [], []
    for i, sym in enumerate(symbols, 1):
        was_cached = (CACHE / f"{sym}.json").exists()

        payload = fetch_chain(sym)
        if payload:
            rows = parse_chain(sym, payload)
            option_rows_all.extend(rows)
            u = parse_underlying(sym, payload, rows)
            if u:
                underlying_rows.append(u)

        if not was_cached:  # only pace real network requests
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            if i % LONG_PAUSE_EVERY == 0:
                time.sleep(random.uniform(*LONG_PAUSE_RANGE))
        if i % 100 == 0:
            print(f"{i}/{len(symbols)} done")
    return pd.DataFrame(option_rows_all), pd.DataFrame(underlying_rows)


def filter_liquid(df: pd.DataFrame) -> pd.DataFrame:
    """Drop contracts with no bid, no IV, low open interest, or wide spreads."""
    df = df.copy()
    for col in ["bid", "ask", "iv", "open_interest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mid = (df["bid"] + df["ask"]) / 2
    spread_pct = (df["ask"] - df["bid"]) / mid
    return df[
        (df["bid"] > 0)
        & (df["iv"] > 0)
        & (df["open_interest"] >= 100)
        & (spread_pct < 0.25)
    ]


if __name__ == "__main__":
    symbols = [s.strip() for s in Path("NEWsymbols.txt").read_text().split() if s.strip()]
    options_df, underlyings_df = sweep(symbols)

    # options_df.to_csv(f"{date_str}/options_snapshot.csv", index=False)
    underlyings_df.to_csv(f"{date_str}/underlyings_snapshot.csv", index=False)

    liquid = filter_liquid(options_df)
    liquid.to_csv(f"{date_str}/options_snapshot.csv", index=False)
    
    print(len(options_df), "contracts,", len(liquid), "after liquidity filter")
    print(len(underlyings_df), "underlyings")

    end_date_str = str(datetime.datetime.now())

    with open(f"{date_str}/DONE", "w") as f:
        f.write(f"FINISHED looking at {len(options_df)} options contracts\n\
                {len(liquid)} left after liquidity filter\n\
                {len(underlyings_df)} underlyings\n\
                Started at {start_date_str}\n\
                Finished at {end_date_str}")

    shutil.make_archive(f"{date_str}/cboe_cache", 'zip', f"{date_str}/cboe_cache")

    # 2. Delete the original directory completely
    shutil.rmtree(f"{date_str}/cboe_cache")