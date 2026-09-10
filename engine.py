"""Student robo-advisor engine.

Pure computation, no UI. Everything needed to run offline is hardcoded here:
long-run capital-market assumptions (expected returns, vols, correlations)
and a 2007-2024 calendar-year total-return matrix for backtests.

Two allocation methods are offered:

  A. Heuristic / lifecycle  -> ``heuristic_allocation``
     Age-based equity share scaled by risk tolerance, then split into fixed
     sleeves. This is the "110 minus age" rule of thumb used by most
     target-date glide paths.

  B. Mean-variance (Markowitz) -> ``mean_variance_allocation``
     Maximise expected return for a risk-tolerance-specific target volatility
     under long-only, fully-invested, sleeve-cap and concentration-penalty
     constraints, solved with scipy's SLSQP.

Modern Portfolio Theory in one paragraph: a portfolio's expected return is the
weighted average of its assets' returns (w . mu), but its variance is
w' Sigma w, which is *less* than the weighted average of the asset variances
whenever correlations are below 1. That gap is the diversification benefit,
and it is why an asset should be judged by its effect on the whole portfolio
(its covariance with everything else), not by its standalone return or risk.

All return / vol inputs are annual, nominal, decimal (0.07 = 7%).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Asset universe and capital-market assumptions
# --------------------------------------------------------------------------

TICKERS = ["VTI", "VXUS", "VWO", "BND", "TIP", "VNQ", "CASH"]

ASSET_NAMES = {
    "VTI": "US Total Stock Market",
    "VXUS": "Intl Developed Stocks",
    "VWO": "Emerging Markets Stocks",
    "BND": "US Aggregate Bonds",
    "TIP": "US TIPS (inflation-linked)",
    "VNQ": "US REITs",
    "CASH": "Cash / T-bills",
}

EQUITY_TICKERS = ["VTI", "VXUS", "VWO", "VNQ"]
BOND_TICKERS = ["BND", "TIP"]

RISK_FREE_RATE = 0.03

# Long-run annual expected (arithmetic) returns and volatilities.
# Broadly in line with published 10-year capital-market assumptions from
# the large asset managers, rounded for classroom use.
EXPECTED_RETURNS = np.array([
    0.070,   # VTI
    0.072,   # VXUS
    0.080,   # VWO
    0.045,   # BND
    0.040,   # TIP
    0.068,   # VNQ
    0.030,   # CASH
])

VOLATILITIES = np.array([
    0.160,   # VTI
    0.170,   # VXUS
    0.220,   # VWO
    0.055,   # BND
    0.060,   # TIP
    0.200,   # VNQ
    0.010,   # CASH
])

# 7x7 correlation matrix, symmetric, ones on the diagonal.
CORRELATION = np.array([
    #  VTI   VXUS   VWO    BND    TIP    VNQ   CASH
    [1.00,  0.85,  0.72,  0.10,  0.05,  0.70,  0.00],   # VTI
    [0.85,  1.00,  0.85,  0.15,  0.10,  0.62,  0.00],   # VXUS
    [0.72,  0.85,  1.00,  0.12,  0.10,  0.55,  0.00],   # VWO
    [0.10,  0.15,  0.12,  1.00,  0.75,  0.25,  0.15],   # BND
    [0.05,  0.10,  0.10,  0.75,  1.00,  0.25,  0.10],   # TIP
    [0.70,  0.62,  0.55,  0.25,  0.25,  1.00,  0.00],   # VNQ
    [0.00,  0.00,  0.00,  0.15,  0.10,  0.00,  1.00],   # CASH
])

# Covariance matrix: Sigma = D R D where D = diag(vols).
COVARIANCE = np.outer(VOLATILITIES, VOLATILITIES) * CORRELATION

# --------------------------------------------------------------------------
# Optional live data (yfinance) with hardcoded fallback
#
# The hardcoded arrays above are the default and are always enough to launch.
# ``use_live_data()`` is opt-in: it tries to download monthly ETF returns and
# annualise them; on ANY failure (yfinance not installed, no network, a
# missing ticker, too little history, NaNs) it leaves the hardcoded numbers
# untouched and returns False. The 2007-2024 backtest matrix below is never
# replaced; it is a fixed historical record, not an estimate.
# --------------------------------------------------------------------------

DATA_SOURCE = "hardcoded long-run assumptions"
LIVE_PROXY = {"CASH": "BIL"}        # cash has no ticker; BIL (1-3m T-bills) stands in
LIVE_START = "2012-01-01"           # VXUS launched Jan 2011; skip its thin first year
MIN_LIVE_MONTHS = 60


def load_live_moments(start: str = LIVE_START, end: str | None = None) -> dict:
    """Download monthly total returns from yfinance and annualise them.

    Returns {"mu", "vol", "corr", "n_months", "start", "end"} with arrays in
    TICKERS order: mu = 12 x mean monthly return, vol = sqrt(12) x monthly
    std, corr = monthly correlation matrix. Raises on any failure so the
    caller can fall back to the hardcoded moments.
    """
    import pandas as pd
    import yfinance as yf

    symbols = [LIVE_PROXY.get(t, t) for t in TICKERS]
    raw = yf.download(symbols, start=start, end=end, interval="1mo",
                      auto_adjust=True, progress=False, threads=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
    prices = prices.reindex(columns=symbols).dropna()
    rets = prices.pct_change().dropna()
    if len(rets) < MIN_LIVE_MONTHS:
        raise RuntimeError(f"only {len(rets)} months of overlapping data "
                           f"(need {MIN_LIVE_MONTHS})")
    mu = rets.mean().to_numpy(dtype=float) * 12.0
    vol = rets.std(ddof=1).to_numpy(dtype=float) * np.sqrt(12.0)
    corr = rets.corr().to_numpy(dtype=float)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(vol))
            and np.all(np.isfinite(corr)) and np.all(vol > 0)):
        raise RuntimeError("non-finite or zero moments in downloaded data")
    return {
        "mu": mu, "vol": vol, "corr": corr, "n_months": int(len(rets)),
        "start": str(rets.index[0].date()), "end": str(rets.index[-1].date()),
    }


def use_live_data(start: str = LIVE_START, end: str | None = None) -> bool:
    """Try to replace the hardcoded moments with yfinance estimates.

    On success the module arrays are updated IN PLACE, so functions whose
    default arguments captured EXPECTED_RETURNS / COVARIANCE at import time
    see the new numbers too, and DATA_SOURCE records the sample window.
    On failure nothing changes, DATA_SOURCE records why, and the app keeps
    running on the hardcoded matrix. Never required for launch.
    """
    global DATA_SOURCE
    try:
        m = load_live_moments(start, end)
    except Exception as exc:  # any failure at all means: stay on hardcoded
        DATA_SOURCE = (f"hardcoded long-run assumptions "
                       f"(live download failed: {type(exc).__name__}: {exc})")
        return False
    EXPECTED_RETURNS[:] = m["mu"]
    VOLATILITIES[:] = m["vol"]
    CORRELATION[:] = m["corr"]
    COVARIANCE[:] = np.outer(VOLATILITIES, VOLATILITIES) * CORRELATION
    DATA_SOURCE = (f"yfinance monthly returns {m['start']} to {m['end']} "
                   f"({m['n_months']} months; cash = BIL)")
    return True

# --------------------------------------------------------------------------
# Historical calendar-year total returns, 2007-2024 (decimal)
# Approximate ETF total returns (VXUS uses FTSE all-world ex-US before its
# 2011 launch; CASH is the 3-month T-bill). Good enough for a backtest that
# is meant to show sequence risk and drawdowns, not to be audited.
# --------------------------------------------------------------------------

HIST_YEARS = list(range(2007, 2025))

HIST_RETURNS = np.array([
    #   VTI     VXUS    VWO     BND     TIP     VNQ     CASH
    [ 0.055,  0.155,  0.390,  0.069,  0.116, -0.164,  0.047],   # 2007
    [-0.369, -0.440, -0.525,  0.051, -0.024, -0.370,  0.016],   # 2008
    [ 0.288,  0.389,  0.763,  0.060,  0.114,  0.301,  0.001],   # 2009
    [ 0.173,  0.118,  0.190,  0.065,  0.061,  0.284,  0.001],   # 2010
    [ 0.010, -0.145, -0.187,  0.077,  0.133,  0.086,  0.000],   # 2011
    [ 0.164,  0.182,  0.188,  0.040,  0.064,  0.176,  0.001],   # 2012
    [ 0.335,  0.151, -0.050, -0.021, -0.085,  0.024,  0.000],   # 2013
    [ 0.126, -0.042,  0.006,  0.059,  0.036,  0.304,  0.000],   # 2014
    [ 0.004, -0.043, -0.154,  0.004, -0.018,  0.024,  0.000],   # 2015
    [ 0.127,  0.047,  0.117,  0.026,  0.047,  0.085,  0.003],   # 2016
    [ 0.212,  0.275,  0.314,  0.036,  0.029,  0.049,  0.009],   # 2017
    [-0.051, -0.144, -0.146, -0.001, -0.014, -0.060,  0.019],   # 2018
    [ 0.308,  0.215,  0.203,  0.087,  0.084,  0.289,  0.021],   # 2019
    [ 0.210,  0.113,  0.152,  0.077,  0.108, -0.047,  0.004],   # 2020
    [ 0.257,  0.086,  0.013, -0.017,  0.057,  0.405,  0.000],   # 2021
    [-0.195, -0.160, -0.178, -0.131, -0.122, -0.262,  0.015],   # 2022
    [ 0.260,  0.155,  0.093,  0.057,  0.038,  0.118,  0.050],   # 2023
    [ 0.238,  0.051,  0.106,  0.014,  0.018,  0.048,  0.052],   # 2024
])

assert HIST_RETURNS.shape == (len(HIST_YEARS), len(TICKERS))

# --------------------------------------------------------------------------
# Risk-tolerance settings
# --------------------------------------------------------------------------

RISK_LEVELS = ["Conservative", "Moderate", "Aggressive"]

# Method A: multiplier on the (110 - age)/100 equity share.
RISK_MULTIPLIER = {"Conservative": 0.72, "Moderate": 1.00, "Aggressive": 1.22}

# Method B: target annual portfolio volatility.
TARGET_VOL = {"Conservative": 0.07, "Moderate": 0.11, "Aggressive": 0.15}

EQUITY_FLOOR, EQUITY_CAP = 0.20, 0.95
CASH_FLOOR = 0.02

# Method A sleeve splits (must sum to 1 within each group).
EQUITY_SPLIT = {"VTI": 0.58, "VXUS": 0.22, "VWO": 0.10, "VNQ": 0.10}
BOND_SPLIT = {"BND": 0.70, "TIP": 0.30}

# Method B: per-asset weight caps so the optimiser cannot collapse into the
# two "corner" assets (highest-Sharpe stock + lowest-vol bond).
SLEEVE_CAPS = {
    "VTI": 0.60, "VXUS": 0.30, "VWO": 0.15,
    "BND": 0.60, "TIP": 0.30, "VNQ": 0.15, "CASH": 0.20,
}
# Method B: penalty on sum(w^2) (Herfindahl). Higher = more diversified.
CONCENTRATION_PENALTY = 0.02

# --------------------------------------------------------------------------
# Client profile
# --------------------------------------------------------------------------


@dataclass
class ClientProfile:
    age: int
    risk: str                    # one of RISK_LEVELS
    horizon_years: int
    initial: float               # starting balance ($)
    monthly_contribution: float  # $ per month
    goal: str = "Retirement"
    annual_income: float = 0.0   # $ per year; wages if working, pension/SS-like if retired

    def __post_init__(self):
        if self.risk not in RISK_LEVELS:
            raise ValueError(f"risk must be one of {RISK_LEVELS}, got {self.risk!r}")
        if not 18 <= self.age <= 100:
            raise ValueError("age must be between 18 and 100")
        if self.horizon_years < 1:
            raise ValueError("horizon_years must be >= 1")
        if self.initial < 0 or self.monthly_contribution < 0:
            raise ValueError("initial and monthly_contribution must be >= 0")
        if self.annual_income < 0:
            raise ValueError("annual_income must be >= 0")


# --------------------------------------------------------------------------
# Portfolio statistics (MPT core)
# --------------------------------------------------------------------------


def _as_weights(w) -> np.ndarray:
    """Accept a dict {ticker: weight} or an array in TICKERS order."""
    if isinstance(w, dict):
        return np.array([float(w.get(t, 0.0)) for t in TICKERS])
    w = np.asarray(w, dtype=float)
    if w.shape != (len(TICKERS),):
        raise ValueError(f"weights must have length {len(TICKERS)}")
    return w


def portfolio_return(w, mu: np.ndarray = EXPECTED_RETURNS) -> float:
    """Expected portfolio return  E[r_p] = w . mu."""
    return float(_as_weights(w) @ mu)


def portfolio_variance(w, cov: np.ndarray = COVARIANCE) -> float:
    """Portfolio variance  sigma_p^2 = w' Sigma w."""
    w = _as_weights(w)
    return float(w @ cov @ w)


def portfolio_vol(w, cov: np.ndarray = COVARIANCE) -> float:
    """Portfolio volatility  sigma_p = sqrt(w' Sigma w)."""
    return float(np.sqrt(max(portfolio_variance(w, cov), 0.0)))


def sharpe_ratio(w, rf: float = RISK_FREE_RATE) -> float:
    """(E[r_p] - rf) / sigma_p."""
    vol = portfolio_vol(w)
    return (portfolio_return(w) - rf) / vol if vol > 0 else 0.0


def equity_share(w) -> float:
    """Combined weight in VTI + VXUS + VWO + VNQ."""
    w = _as_weights(w)
    return float(sum(w[TICKERS.index(t)] for t in EQUITY_TICKERS))


def diversification_benefit(w) -> float:
    """Weighted-average standalone vol minus actual portfolio vol.

    Positive whenever correlations < 1. This is the MPT payoff in one number.
    """
    w = _as_weights(w)
    return float(w @ VOLATILITIES - portfolio_vol(w))


def weights_to_dict(w) -> dict:
    w = _as_weights(w)
    return {t: float(x) for t, x in zip(TICKERS, w)}


def portfolio_stats(w) -> dict:
    return {
        "expected_return": portfolio_return(w),
        "volatility": portfolio_vol(w),
        "sharpe": sharpe_ratio(w),
        "equity_share": equity_share(w),
        "diversification_benefit": diversification_benefit(w),
    }


# --------------------------------------------------------------------------
# Method A: heuristic / lifecycle
# --------------------------------------------------------------------------


def heuristic_equity_share(age: int, risk: str) -> float:
    raw = (110 - age) / 100.0 * RISK_MULTIPLIER[risk]
    return float(np.clip(raw, EQUITY_FLOOR, EQUITY_CAP))


def heuristic_allocation(age: int, risk: str) -> np.ndarray:
    """Lifecycle rule: equity = clip((110 - age)/100 * risk_mult, 0.20, 0.95).

    Equity is split 58/22/10/10 across VTI/VXUS/VWO/VNQ. What is left goes to
    a fixed-income sleeve: a small cash floor first, then 70/30 BND/TIP.
    """
    eq = heuristic_equity_share(age, risk)
    fixed = 1.0 - eq
    cash = min(CASH_FLOOR, fixed)
    bonds = fixed - cash

    w = np.zeros(len(TICKERS))
    for t, s in EQUITY_SPLIT.items():
        w[TICKERS.index(t)] = eq * s
    for t, s in BOND_SPLIT.items():
        w[TICKERS.index(t)] = bonds * s
    w[TICKERS.index("CASH")] = cash

    w = np.clip(w, 0.0, None)
    return w / w.sum()


# --------------------------------------------------------------------------
# Method B: mean-variance optimisation (Markowitz, SLSQP)
# --------------------------------------------------------------------------

_LONG_ONLY_BOUNDS = [(0.0, 1.0)] * len(TICKERS)
_FULLY_INVESTED = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _solve(objective, constraints, bounds, x0=None) -> np.ndarray:
    if x0 is None:
        x0 = np.full(len(TICKERS), 1.0 / len(TICKERS))
    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def mean_variance_allocation(
    risk: str,
    target_vol: float | None = None,
    caps: dict | None = SLEEVE_CAPS,
    penalty: float = CONCENTRATION_PENALTY,
) -> np.ndarray:
    """Maximise  w.mu - penalty * sum(w^2)  subject to
         sum(w) = 1,  w >= 0 (no shorting),  sqrt(w' Sigma w) <= target_vol,
         and w_i <= cap_i for each sleeve.

    The Herfindahl penalty (sum of squared weights) and the sleeve caps keep
    the solution from collapsing into a VTI + BND barbell, which is what an
    unconstrained optimiser does with almost any set of inputs.
    """
    if target_vol is None:
        target_vol = TARGET_VOL[risk]

    def objective(w):
        return -(w @ EXPECTED_RETURNS) + penalty * float(w @ w)

    constraints = [
        _FULLY_INVESTED,
        {"type": "ineq", "fun": lambda w: target_vol**2 - w @ COVARIANCE @ w},
    ]
    bounds = [
        (0.0, caps.get(t, 1.0) if caps else 1.0) for t in TICKERS
    ]
    return _solve(objective, constraints, bounds)


def min_variance_portfolio(caps: dict | None = None) -> np.ndarray:
    """Global minimum-variance portfolio, long-only."""
    bounds = [(0.0, caps.get(t, 1.0) if caps else 1.0) for t in TICKERS]
    return _solve(lambda w: w @ COVARIANCE @ w, [_FULLY_INVESTED], bounds)


def efficient_frontier(n_points: int = 30, caps: dict | None = None) -> dict:
    """Long-only efficient frontier.

    For a grid of target returns between the min-variance portfolio's return
    and the highest achievable return, solve
        min  w' Sigma w   s.t.  w.mu = target,  sum(w) = 1,  w >= 0.
    Returns arrays of returns, vols, Sharpe ratios and the weight matrix.
    """
    bounds = [(0.0, caps.get(t, 1.0) if caps else 1.0) for t in TICKERS]
    w_min = min_variance_portfolio(caps)
    r_lo = portfolio_return(w_min)

    # Highest return under the same bounds (linear, so SLSQP handles it).
    w_max = _solve(lambda w: -(w @ EXPECTED_RETURNS), [_FULLY_INVESTED], bounds)
    r_hi = portfolio_return(w_max)

    targets = np.linspace(r_lo, r_hi, n_points)
    rets, vols, weights = [], [], []
    x0 = w_min
    for tr in targets:
        cons = [
            _FULLY_INVESTED,
            {"type": "eq", "fun": lambda w, tr=tr: w @ EXPECTED_RETURNS - tr},
        ]
        w = _solve(lambda w: w @ COVARIANCE @ w, cons, bounds, x0=x0)
        x0 = w
        rets.append(portfolio_return(w))
        vols.append(portfolio_vol(w))
        weights.append(w)

    rets, vols = np.array(rets), np.array(vols)
    return {
        "returns": rets,
        "vols": vols,
        "sharpes": (rets - RISK_FREE_RATE) / vols,
        "weights": np.array(weights),
        "min_variance": w_min,
    }


# --------------------------------------------------------------------------
# Backtest on the 2007-2024 annual matrix
# --------------------------------------------------------------------------


def max_drawdown(path: np.ndarray) -> float:
    """Largest peak-to-trough decline of a wealth path, as a negative decimal."""
    path = np.asarray(path, dtype=float)
    peaks = np.maximum.accumulate(path)
    return float(np.min(path / peaks - 1.0))


def backtest(w, initial: float = 10_000.0) -> dict:
    """Grow `initial` through 2007-2024 with annual rebalancing to `w`.

    Returns the wealth path (year-end values, with the starting value first),
    the annual portfolio returns, CAGR, realised vol and max drawdown.
    """
    w = _as_weights(w)
    port_rets = HIST_RETURNS @ w                        # one number per year
    path = initial * np.cumprod(np.concatenate([[1.0], 1.0 + port_rets]))
    n = len(port_rets)
    cagr = (path[-1] / path[0]) ** (1.0 / n) - 1.0
    return {
        "years": [HIST_YEARS[0] - 1] + HIST_YEARS,     # 2006 = start
        "path": path,
        "annual_returns": port_rets,
        "cagr": float(cagr),
        "realised_vol": float(np.std(port_rets, ddof=1)),
        "max_drawdown": max_drawdown(path),
        "worst_year": (HIST_YEARS[int(np.argmin(port_rets))], float(port_rets.min())),
        "best_year": (HIST_YEARS[int(np.argmax(port_rets))], float(port_rets.max())),
        "final_value": float(path[-1]),
    }


# --------------------------------------------------------------------------
# Monte Carlo wealth projection
# --------------------------------------------------------------------------


def monte_carlo(
    w,
    initial: float,
    monthly_contribution: float,
    years: int,
    n_paths: int = 1000,
    seed: int = 42,
) -> dict:
    """Simulate wealth with monthly multivariate-normal asset returns.

    Monthly mean = mu/12, monthly covariance = Sigma/12. The portfolio is
    rebalanced to `w` every month (constant mix), so each month's portfolio
    return is w . r_month. Contributions are added at the end of each month.
    Returns mean / P25 / P75 (and a few more percentiles) of the terminal
    wealth, plus the same percentiles along the path for plotting.
    """
    w = _as_weights(w)
    rng = np.random.default_rng(seed)
    months = int(years * 12)

    mu_m = EXPECTED_RETURNS / 12.0
    cov_m = COVARIANCE / 12.0
    L = np.linalg.cholesky(cov_m)

    # z: (n_paths, months, n_assets) standard normals -> correlated returns
    # r = mu + L z. einsum rather than @: the batched matmul path on macOS
    # Accelerate BLAS raises spurious floating-point warnings for this shape.
    z = rng.standard_normal((n_paths, months, len(TICKERS)))
    asset_rets = mu_m + np.einsum("pmk,jk->pmj", z, L)
    port_rets = np.maximum(np.einsum("pmj,j->pm", asset_rets, w), -0.99)

    wealth = np.empty((n_paths, months + 1))
    wealth[:, 0] = initial
    for m in range(months):
        wealth[:, m + 1] = wealth[:, m] * (1.0 + port_rets[:, m]) + monthly_contribution

    terminal = wealth[:, -1]
    total_contrib = initial + monthly_contribution * months
    pct = lambda q: float(np.percentile(terminal, q))
    return {
        "mean": float(terminal.mean()),
        "median": pct(50),
        "p5": pct(5),
        "p25": pct(25),
        "p75": pct(75),
        "p95": pct(95),
        "prob_loss": float(np.mean(terminal < total_contrib)),
        "total_contributions": total_contrib,
        "n_paths": n_paths,
        "months": months,
        "terminal": terminal,
        "path_percentiles": {
            q: np.percentile(wealth, q, axis=0) for q in (5, 25, 50, 75, 95)
        },
        "path_mean": wealth.mean(axis=0),
    }


# --------------------------------------------------------------------------
# Method C: research-informed  (enabled via RESEARCH_MODE_ENABLED)
#   Duarte, Fonseca, Goodman & Parker (NBER 29559, 2021): the optimal equity
#   share of financial wealth is hump-shaped, ~80% near age 45, settling at a
#   flat ~60% in retirement. Age-only TDF rules that fall to 30-40% by the
#   late 60s cost ~2-3% of consumption.
#   Choi (JEP 2022): human capital belongs on the balance sheet; it behaves
#   like a bond, so financial wealth should carry more equity when H/W is high.
#   Choi, Liu & Liu (2025, "Practical Finance"):
#       alpha* = clip((r - rf) / (gamma * sigma^2), 0, 1)
#       alpha^ = clip(alpha* * (1 + H / W), 0, 1)
#   Mode C blends the Duarte glide with the Choi tilt, then splits the equity
#   share with the same sleeve rule as Mode A.
# --------------------------------------------------------------------------

RESEARCH_MODE_ENABLED = True    # False hides Mode C from recommend() and the UI

RETIREMENT_AGE = 67
LIFE_EXPECTANCY_AGE = 90
PENSION_REPLACEMENT = 0.40      # pension/SS-like income after 67, share of pay
HC_DISCOUNT_RATE = RISK_FREE_RATE

# Income stand-ins used only when the client enters $0 income (Choi, Liu &
# Liu: never let a retiree's H collapse to zero; SS/pension income exists).
RETIREE_INCOME_DRAW = 0.04      # pension/SS stand-in = max(4% of W, floor)
RETIREE_INCOME_FLOOR = 24_000.0
WORKER_SAVINGS_RATE = 0.15      # worker stand-in = 12 * monthly / 15%, floored
WORKER_INCOME_FLOOR = 40_000.0

DUARTE_AGES = np.array([25.0, 45.0, 67.0])
DUARTE_EQUITY = np.array([0.88, 0.80, 0.60])   # flat 60% at and after 67

CHOI_GAMMA = 4.0                # moderate risk aversion; the paper's gamma=7
                                # example is deliberately NOT used on its own
RESEARCH_BLEND_DUARTE = 0.60    # 60% Duarte glide + 40% Choi raw share
RESEARCH_TILT = {"Conservative": -0.05, "Moderate": 0.0, "Aggressive": 0.05}
HORIZON_CAPPED_GOALS = {"Home purchase", "Education"}


def annuity_factor(n_years: float, rate: float = HC_DISCOUNT_RATE) -> float:
    """PV of $1 per year for n_years, paid at year end."""
    n = max(float(n_years), 0.0)
    if n == 0.0:
        return 0.0
    if rate == 0.0:
        return n
    return (1.0 - (1.0 + rate) ** -n) / rate


def is_retired(age: int) -> bool:
    return age >= RETIREMENT_AGE


def human_capital(age: int, annual_income: float) -> float:
    """Choi's H: PV of remaining labour income plus pension/SS-like income.

    Working client: wages until RETIREMENT_AGE, then PENSION_REPLACEMENT of
    wages until LIFE_EXPECTANCY_AGE. Retired client: the income entered is
    already pension/SS-like and runs to LIFE_EXPECTANCY_AGE.
    """
    if annual_income <= 0:
        return 0.0
    if is_retired(age):
        return annual_income * annuity_factor(LIFE_EXPECTANCY_AGE - age)
    work_years = RETIREMENT_AGE - age
    labour = annual_income * annuity_factor(work_years)
    pension = (PENSION_REPLACEMENT * annual_income
               * annuity_factor(LIFE_EXPECTANCY_AGE - RETIREMENT_AGE)
               * (1.0 + HC_DISCOUNT_RATE) ** -work_years)
    return labour + pension


def inferred_income(client: ClientProfile) -> tuple[float, bool]:
    """Annual income to use for H, and whether it was inferred.

    Entered income wins when positive. Otherwise a retiree gets a pension /
    Social Security stand-in of max(4% x W, $24k), and a worker gets
    12 x monthly / 15% savings rate, floored at $40k. Retiree income is
    therefore never $0.
    """
    if client.annual_income > 0:
        return float(client.annual_income), False
    if is_retired(client.age):
        return max(RETIREE_INCOME_DRAW * client.initial, RETIREE_INCOME_FLOOR), True
    return max(12.0 * client.monthly_contribution / WORKER_SAVINGS_RATE,
               WORKER_INCOME_FLOOR), True


def duarte_glide(age: int, goal: str = "Retirement", horizon_years: int = 30) -> float:
    """Hump-shaped lifecycle equity share: 88% at 25, 80% at 45, 60% at 67+."""
    eq = float(np.interp(age, DUARTE_AGES, DUARTE_EQUITY))
    if goal in HORIZON_CAPPED_GOALS:
        # A dated liability (house, tuition) shortens the effective horizon.
        eq = min(eq, float(np.clip(0.20 + 0.03 * horizon_years, 0.20, 0.88)))
    return eq


def equity_sleeve_stats() -> tuple[float, float]:
    """Expected return and vol of the 58/22/10/10 equity sleeve."""
    w = np.zeros(len(TICKERS))
    for t, sh in EQUITY_SPLIT.items():
        w[TICKERS.index(t)] = sh
    return portfolio_return(w), portfolio_vol(w)


def choi_equity_share(H: float, W: float, gamma: float = CHOI_GAMMA) -> tuple[float, float]:
    """Return (alpha*, alpha^) from Choi, Liu & Liu (2025)."""
    r, sigma = equity_sleeve_stats()
    alpha_star = float(np.clip((r - RISK_FREE_RATE) / (gamma * sigma ** 2), 0.0, 1.0))
    alpha_hat = float(np.clip(alpha_star * (1.0 + H / max(W, 1.0)), 0.0, 1.0))
    return alpha_star, alpha_hat


def research_equity_share(client: ClientProfile) -> dict:
    """Mode C equity share with every intermediate number, for the UI narrative."""
    duarte = duarte_glide(client.age, client.goal, client.horizon_years)
    income, income_inferred = inferred_income(client)
    H = human_capital(client.age, income)
    W = max(client.initial, 1.0)
    alpha_star, alpha_hat = choi_equity_share(H, W)
    blend = RESEARCH_BLEND_DUARTE * duarte + (1.0 - RESEARCH_BLEND_DUARTE) * alpha_hat
    tilt = RESEARCH_TILT[client.risk]
    equity = float(np.clip(blend + tilt, EQUITY_FLOOR, EQUITY_CAP))
    return {
        "equity": equity,
        "duarte": duarte,
        "H": H,
        "W": W,
        "H_over_W": H / W,
        "alpha_star": alpha_star,
        "alpha_hat": alpha_hat,
        "gamma": CHOI_GAMMA,
        "blend": blend,
        "tilt": tilt,
        "retired": is_retired(client.age),
        "income": income,
        "income_inferred": income_inferred,
        "popular_equity": heuristic_equity_share(client.age, client.risk),
    }


def sleeve_allocation(equity: float) -> np.ndarray:
    """Split an equity share into the seven sleeves with Mode A's rule:
    equity 58/22/10/10 VTI/VXUS/VWO/VNQ, 2% cash floor, rest 70/30 BND/TIP."""
    fixed = 1.0 - equity
    cash = min(CASH_FLOOR, fixed)
    bonds = fixed - cash
    w = np.zeros(len(TICKERS))
    for t, sh in EQUITY_SPLIT.items():
        w[TICKERS.index(t)] = equity * sh
    for t, sh in BOND_SPLIT.items():
        w[TICKERS.index(t)] = bonds * sh
    w[TICKERS.index("CASH")] = cash
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def research_allocation(client: ClientProfile) -> np.ndarray:
    return sleeve_allocation(research_equity_share(client)["equity"])


# --------------------------------------------------------------------------
# One-call recommendation
# --------------------------------------------------------------------------


@dataclass
class Recommendation:
    client: ClientProfile
    method: str
    weights: np.ndarray
    stats: dict = field(default_factory=dict)
    backtest: dict = field(default_factory=dict)
    monte_carlo: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)   # Mode C intermediates

    @property
    def weights_dict(self) -> dict:
        return weights_to_dict(self.weights)


def public_methods() -> list[str]:
    """Allocation methods the UI is allowed to offer."""
    methods = ["heuristic", "mean_variance"]
    if RESEARCH_MODE_ENABLED:
        methods.append("research")
    return methods


def recommend(client: ClientProfile, method: str = "heuristic",
              n_paths: int = 1000, seed: int = 42) -> Recommendation:
    """Run the full pipeline for one client with one allocation method.

    Public entry points: 'heuristic' -> heuristic_allocation,
    'mean_variance' -> mean_variance_allocation. 'research' is refused
    while RESEARCH_MODE_ENABLED is False.
    """
    details = {}
    if method == "heuristic":
        w = heuristic_allocation(client.age, client.risk)
    elif method == "mean_variance":
        w = mean_variance_allocation(client.risk)
    elif method == "research" and RESEARCH_MODE_ENABLED:
        details = research_equity_share(client)
        w = sleeve_allocation(details["equity"])
    else:
        raise ValueError(f"method must be one of {public_methods()}, got {method!r}")

    return Recommendation(
        client=client,
        method=method,
        weights=w,
        stats=portfolio_stats(w),
        backtest=backtest(w, initial=client.initial or 10_000.0),
        monte_carlo=monte_carlo(
            w, client.initial, client.monthly_contribution,
            client.horizon_years, n_paths=n_paths, seed=seed,
        ),
        details=details,
    )


# --------------------------------------------------------------------------
# Console report
# --------------------------------------------------------------------------


def _fmt_weights(w) -> str:
    return "  ".join(f"{t}={x:5.1%}" for t, x in weights_to_dict(w).items())


def print_recommendation(rec: Recommendation) -> None:
    c, s, bt, mc = rec.client, rec.stats, rec.backtest, rec.monte_carlo
    print(f"\n--- {rec.method.upper()} ---")
    print("Weights:      " + _fmt_weights(rec.weights))
    if rec.details:
        d = rec.details
        print(f"Mode C:       Duarte glide {d['duarte']:.1%}  |  H=${d['H']:,.0f}, W=${d['W']:,.0f}, "
              f"H/W={d['H_over_W']:.2f}, alpha*={d['alpha_star']:.1%}, alpha^={d['alpha_hat']:.1%}  |  "
              f"blend {d['blend']:.1%} + tilt {d['tilt']:+.0%} -> equity {d['equity']:.1%}  "
              f"(110-age rule: {d['popular_equity']:.1%})")
    print(f"Equity share: {s['equity_share']:.1%}")
    print(f"Exp. return:  {s['expected_return']:.2%}   Vol: {s['volatility']:.2%}"
          f"   Sharpe (rf={RISK_FREE_RATE:.0%}): {s['sharpe']:.2f}"
          f"   Divers. benefit: {s['diversification_benefit']:.2%}")
    print(f"Backtest 2007-2024 (annual rebalance, start ${bt['path'][0]:,.0f}): "
          f"end ${bt['final_value']:,.0f}, CAGR {bt['cagr']:.2%}, "
          f"realised vol {bt['realised_vol']:.2%}, max DD {bt['max_drawdown']:.1%}, "
          f"worst {bt['worst_year'][0]} {bt['worst_year'][1]:.1%}")
    print(f"Monte Carlo {mc['n_paths']} paths, {c.horizon_years}y, "
          f"${c.initial:,.0f} + ${c.monthly_contribution:,.0f}/mo "
          f"(contributed ${mc['total_contributions']:,.0f}):")
    print(f"   mean ${mc['mean']:,.0f}   P25 ${mc['p25']:,.0f}   "
          f"median ${mc['median']:,.0f}   P75 ${mc['p75']:,.0f}   "
          f"P(< contributions) {mc['prob_loss']:.1%}")


def print_client_report(client: ClientProfile) -> None:
    print("=" * 96)
    print(f"Client: age {client.age}, {client.risk}, {client.horizon_years}y horizon, "
          f"${client.initial:,.0f} initial, ${client.monthly_contribution:,.0f}/mo, "
          f"goal: {client.goal}, income ${client.annual_income:,.0f}/yr")
    print("=" * 96)
    for method in public_methods():
        print_recommendation(recommend(client, method))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print(f"Market assumptions: {DATA_SOURCE}")

    eig = np.linalg.eigvalsh(CORRELATION)
    print(f"Correlation matrix min eigenvalue: {eig.min():.4f} (positive => valid)")

    print("\nStandalone assets:")
    for i, t in enumerate(TICKERS):
        sh = (EXPECTED_RETURNS[i] - RISK_FREE_RATE) / VOLATILITIES[i]
        print(f"  {t:5s} {ASSET_NAMES[t]:28s} E[r]={EXPECTED_RETURNS[i]:.1%}  "
              f"vol={VOLATILITIES[i]:.1%}  Sharpe={sh:.2f}")

    ef = efficient_frontier(n_points=12)
    print("\nLong-only efficient frontier (12 points):")
    print("   E[r]     vol    Sharpe   " + "  ".join(f"{t:>5s}" for t in TICKERS))
    for r, v, sh, w in zip(ef["returns"], ef["vols"], ef["sharpes"], ef["weights"]):
        print(f"  {r:6.2%}  {v:6.2%}   {sh:5.2f}   "
              + "  ".join(f"{x:5.1%}" for x in w))

    print("\nMean-variance targets by risk level:")
    for risk in RISK_LEVELS:
        w = mean_variance_allocation(risk)
        print(f"  {risk:12s} target vol {TARGET_VOL[risk]:.0%}: " + _fmt_weights(w))

    client_a = ClientProfile(age=35, risk="Moderate", horizon_years=30,
                             initial=100_000, monthly_contribution=2_000,
                             goal="Retirement", annual_income=140_000)
    client_b = ClientProfile(age=68, risk="Moderate", horizon_years=20,
                             initial=1_500_000, monthly_contribution=0,
                             goal="Retirement", annual_income=48_000)
    print()
    print_client_report(client_a)
    print()
    print_client_report(client_b)
