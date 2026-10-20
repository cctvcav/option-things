import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import pandas_datareader.data as web

def pred_opt_price(opt: str, type: str, S: float, K: float, iv: float, date: dt.date, exp: dt.date) -> float:
    return -1

def bs_price(flag, S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    put  = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(flag == "c", call, put)

def risk_free_interp(date: dt.date, exp:dt.date) -> float:

    # 1. Get the time until expiration of option in years
    time_til_exp = float((exp - date).days) / 365

    if time_til_exp <= 0:
        raise RuntimeError(f"Expiration date is before the given date")

    # 2. Get the latest Treasury curve (as of your valuation date)
    series = {"DGS1MO": 1/12, "DGS3MO": 3/12, "DGS6MO": 6/12, "DGS1": 1, "DGS2": 2}
    raw = web.DataReader(list(series), "fred", start=f"{date.year}-{date.month}-{date.day-7}").ffill()
    curve = raw.iloc[-1] / 100

    tenors = np.array(list(series.values()))
    yields = curve[list(series)].values

    # Convert to continuously compounded (what Black-Scholes assumes)
    yields_cc = np.log(1 + yields)

    

    # 3. Interpolate a rate for each row
    return np.interp(time_til_exp, tenors, yields_cc)

if __name__ == "__main__":
    date = dt.date(2026, 10, 10)
    exp = dt.date(2028, 11, 20)
    rsrate = risk_free_interp(date, exp)
    print(rsrate)