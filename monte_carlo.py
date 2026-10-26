import numpy as np
import pandas as pd
import datetime as dt
import pandas_market_calendars as mcal

rng = np.random.default_rng()
nyse = mcal.get_calendar("NYSE")



def monte_carlo(base: float, sigma: float, periods: int, num_sims: int = 1000, way: str = "normal") -> np.ndarray:
    sims = np.zeros((num_sims, periods + 1))
    for sim in range(num_sims):
        sims[sim][0] = base
        if way == "normal":
            draws = rng.normal(loc=1, scale=sigma, size=periods)
        for pd in range(1, periods+1):
            sims[sim][pd] = sims[sim][pd-1] * draws
    return sims

def stock_mc(S: float, sigma: float, s_date: dt.date, e_date: dt.date, num_sims: int = 1000, way: str = "normal") -> np.ndarray:
    adj_sigma = sigma / np.sqrt(252)
    period = nyse.valid_days(s_date, e_date)
    gap_days = pd.Series(period).diff().dt.days.fillna(1).to_numpy()
    step_vol = adj_sigma * np.where(gap_days > 1, 1.3, 1.0)
    sims = np.zeros((num_sims, len(period)))
    for sim in range(num_sims):
        sims[sim][0] = S
        for pd in range(1, len(period)):
            draw = 0
            if way == "normal":
                draw = rng.normal(loc=1, scale=step_vol[pd])
            sims[sim][pd] = sims[sim][pd-1] * draw
    return sims