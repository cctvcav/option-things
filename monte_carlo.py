import numpy as np
import pandas as pd

rng = np.random.default_rng()

def monte_carlo_sim(base: float, sigma: float, periods: int, num_sims: int = 1000, way: str = "normal") -> np.ndarray:
    sims = np.zeros((num_sims, periods + 1))
    for sim in range(num_sims):
        periods[sim][0] = base
        draws = rng.normal(loc=1, scale=sigma, size=periods)
        for pd in range(1, periods+1):
            sims[sim][pd] = sims[sim][pd-1] * draws
    return sims