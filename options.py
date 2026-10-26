import datetime as dt
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import pandas_datareader.data as web

today = dt.date.today()

percents = [-5, -4, -3, -2, -1, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.15, -.125, -.1, -.05, -.025, -.01,\
            0, .01, .025, .05, .1, .125, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5]

percents = [-.05, -.025, -.01,\
            0, .01, .025, .05]

percents = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

series = {"DGS1MO": 1/12, "DGS3MO": 3/12, "DGS6MO": 6/12, "DGS1": 1, "DGS2": 2}
raw = web.DataReader(list(series), "fred", start=f"{today.year}-{today.month}-{today.day-12}").ffill()

# returns a dictionary that has the price of the underlying move according to its iv in a given time.
# accounts for underlying moving up and down
def pred_opt_price(opt: str, flag: str, S: float, K: float, comp_iv: float, opt_iv: float, s_date: dt.date, e_date: dt.date, exp: dt.date) -> dict[float, float]:
    time_delta = years_til(s_date, e_date)
    time_til_exp = years_til(e_date, exp)
    print("time_to_exp:", time_til_exp)
    adj_iv = comp_iv * np.sqrt(time_delta)
    S_up = (1 + adj_iv) * S
    S_down = (1 - adj_iv) * S
    r = risk_free_interp(e_date, exp)
    price_if_S_up = bs_price(flag, S_up, K, time_til_exp, r, opt_iv)
    price_if_S_down = bs_price(flag, S_down, K, time_til_exp, r, opt_iv)

    return {S_up:price_if_S_up, S_down:price_if_S_down}

def pred_opt_mvt(opt: str, flag: str, S: float, K: float, comp_iv: float, opt_iv: float, s_date: dt.date, e_date: dt.date, exp: dt.date, way: str = "basic") -> dict[tuple[float, float], float]:
    res = {}
    sec_mvt = pred_sec_mvt(opt, S, comp_iv, s_date, e_date, way)
    # print(sec_mvt)
    for interval in sec_mvt:
        low = interval[0]
        high = interval[1]
        percent = sec_mvt[interval]
        if low <= 0 or high <= 0:
            continue
        low_bs = bs_price(flag, low, K, years_til(e_date, exp), risk_free_interp(e_date, exp), opt_iv)
        high_bs = bs_price(flag, high, K, years_til(e_date, exp), risk_free_interp(e_date, exp), opt_iv)
        res[(float(low_bs), float(high_bs))] = percent * 100
    return res

def make_mvt_graph(opt: str, flag: str, K: float, s_date: dt.date, e_date: dt.date, exp: dt.date, percent_changes: dict[tuple[float, float], float]) -> None:
    key_list = list(percent_changes.keys())
    edges = np.array([round(a, 2) for a, _ in key_list] + [round(key_list[-1][1], 2)])
    # print(edges)
    values = np.array([val for val in percent_changes.values()])
    pos = np.arange(len(values))
    plt.bar(pos, values, width=1, align="edge", edgecolor="black")
    plt.xticks(np.arange(len(edges)), edges)
    plt.title(f"{opt} {K} {flag} {exp.month}-{exp.day}-{exp.year}: {s_date.month}-{s_date.day}-{s_date.year} to {e_date.month}-{e_date.day}-{e_date.year}")
    plt.xlabel("Option Price")
    plt.ylabel("% Chance")
    plt.savefig(f"{opt}_{K}_{flag}_{exp.month}-{exp.day}-{exp.year}_from_{s_date.month}-{s_date.day}-{s_date.year}_to_{e_date.month}-{e_date.day}-{e_date.year}.png")
    plt.clf()

def pred_sec_mvt(opt: str, S: float, iv: float, s_date: dt.date, e_date: dt.date, way: str = "basic") -> dict[tuple[float, float], float]:
    # today = dt.date.today()
    time_delta = years_til(s_date, e_date)
    if way == "basic":
        return basic_pred(S, iv, time_delta)

    return {}

def basic_pred(S: float, iv: float, time_delta: dt.date) -> dict[tuple[float, float], float]:
    adj_iv = iv * np.sqrt(time_delta)
    res = {}
    for i in range(len(percents) - 1):
        start, end = (percents[i] * adj_iv)+1, (percents[i+1] * adj_iv)+1
        res[(start*S, end*S)] = norm.cdf(x=end*S, loc=S, scale=adj_iv*S) - norm.cdf(x=start*S, loc=S, scale=adj_iv*S)
    return res

def bs_price(flag, S, K, t, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    call = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    put  = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(flag == "c", call, put)

def risk_free_interp(date: dt.date, exp:dt.date) -> float:

    # 1. Get the time until expiration of option in years
    time_til_exp = years_til(date, exp)

    if time_til_exp <= 0:
        raise RuntimeError(f"Expiration date is before the given date")

    # 2. Get the latest Treasury curve (as of your valuation date)
    curve = raw.iloc[-1] / 100

    tenors = np.array(list(series.values()))
    yields = curve[list(series)].values

    # Convert to continuously compounded (what Black-Scholes assumes)
    yields_cc = np.log(1 + yields)

    # 3. Interpolate a rate for each row
    return np.interp(time_til_exp, tenors, yields_cc)

def years_til(now: dt.date, later:dt.date) -> float:
    return float((later - now).days) / 365

def get_gain_loss(up: float, down: float, P: float, flag: str = "c",) -> tuple[float, float]:
    if flag == "c":
        return (up - P, P - down)
    else:
        return (down - P, P - up)


if __name__ == "__main__":
    # date = dt.date(2026, 10, 10)
    # exp = dt.date(2028, 11, 20)
    # rsrate = risk_free_interp(date, exp)
    # print(rsrate)
    commands = set(["c", "p", "s", "o", "TESTC"])
    while(True):
        print("What do you want to do? (Enter one of the following)")
        print("c: Compare underlying movement up to underlying movement down on an option")
        print("p: Price an option")
        print("s: Predict stock movement")
        print("o: Predict option movement")
        user_input = input()
        if not user_input in commands:
            print(f"{user_input} is not an option.")
        if user_input == "c":
            opt = input("Enter a stock symbol:")
            while(True):
                flag = input("Enter \"c\" for call or \"p\" for put:").lower()
                if flag == "c" or flag == "p":
                    break
            S = float(input("Enter the stock price:"))
            K = float(input("Enter the strike price:"))
            P = float(input("Enter the option price:"))
            comp_iv = float(input("Enter the stock's implied volatility (in percent):")) / 100
            opt_iv = float(input("Enter the individual option's implied volatility (in percent): ")) / 100
            date_choice = input("Do you want to use todays date for the initial date? [y/n]:")
            s_date = today
            if date_choice == "n":
                y = input("Enter the year YYYY:")
                m = input("Enter the month MM:")
                d = input("Enter the day DD:")
                s_date = dt.date(int(y), int(m), int(d))
            print("Enter the date you want to find the option prices of.\nMake sure it's after the initial date")
            y = input("Enter the year YYYY:")
            m = input("Enter the month MM:")
            d = input("Enter the day DD:")
            e_date = dt.date(int(y), int(m), int(d))
            print("Enter the option's expiration date")
            exp_y = input("Enter the year YYYY:")
            exp_m = input("Enter the month MM:")
            exp_d = input("Enter the day DD:")
            exp = dt.date(int(exp_y), int(exp_m), int(exp_d))
            opt_preds = pred_opt_price(opt, flag, S, K, comp_iv, opt_iv, s_date, e_date, exp)
            S_up = max(opt_preds.keys())
            S_down = min(opt_preds.keys())
            gain, loss = get_gain_loss(opt_preds[S_up], opt_preds[S_down], P, flag)
            print(f"Underlying 1-week up: {S_up}\nUnderlying 1-week down: {S_down}\nGain: {gain}\nLoss: {loss}\nGain:Loss: {gain/loss}")
        if user_input == "p" or user_input == "o":
            if user_input == "o":
                opt = input("Enter a stock symbol:")
            while(True):
                flag = input("Enter \"c\" for call or \"p\" for put: ")
                if flag == "c" or flag == "p":
                    break
                print(f"{flag} is not \"c\" or \"p\"")
            S = float(input("Enter the stock price: "))
            K = float(input("Enter the strike price: "))
            date_choice = input("Do you want to use todays date for the initial date? [y/n]:")
            date = today
            if date_choice == "n":
                print("Enter the date you want to find the option price of.")
                y = input("Enter the year YYYY:")
                m = input("Enter the month MM:")
                d = input("Enter the day DD:")
                date = dt.date(int(y), int(m), int(d))
            print("Enter the option's expiration date")
            exp_y = input("Enter the year YYYY:")
            exp_m = input("Enter the month MM:")
            exp_d = input("Enter the day DD:")
            exp = dt.date(int(exp_y), int(exp_m), int(exp_d))
            t = (exp - date).days / 365
            r = risk_free_interp(date, exp)
            r_choice = input("Do you want to automatically find a risk-free rate? [y/n]: ")
            if r_choice == "n":
                r = float(input("Enter a risk-free rate (in percent): ")) / 100
            if user_input == "o":
                comp_iv = float(input("Enter the stock's/composite implied volatility (in percent): ")) / 100
                opt_iv = float(input("Enter the individual option's implied volatility (in percent): ")) / 100
                way = input("Enter the way you want the stock price to be distributed [basic/fattail/lognormal]: ")
                percents = pred_opt_mvt(opt, flag, S, K, comp_iv, opt_iv, date, exp, way)
                make_mvt_graph(opt, flag, K, date, exp, percents)
            if user_input == "p":
                sigma = float(input("Enter the option's implied volatility (in percent): ")) / 100
                opt_price = bs_price(flag, S, K, t, r, sigma)
                print(f"Theoretical option price: {opt_price}")



        break