---
title: Olin Robo-Advisor
emoji: 📈
sdk: gradio
app_file: app.py
---

# Olin Robo-Advisor

Educational prototype. Not investment advice.

A student robo-advisor that builds a seven-ETF portfolio (VTI, VXUS, VWO, BND, TIP, VNQ, cash) from a client profile and compares three allocation engines:

| Mode | Engine | Rule |
|---|---|---|
| A | Heuristic (110 − age) | equity = clip((110 − age)/100 × risk multiplier, 20%, 95%) |
| B | Mean-variance optimization | maximise return at a risk-level target volatility, long-only, sleeve caps, SLSQP |
| C | Research-informed | Duarte et al. (2021) lifecycle glide blended with a Choi (2022; Choi, Liu & Liu 2025) human-capital tilt |

All expected returns, volatilities, correlations, and the 2007–2024 annual return matrix are hardcoded in `engine.py`, so the app runs offline.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio Blocks UI. Validates inputs, calls `engine.py`, draws the Plotly charts. No portfolio math. |
| `engine.py` | All the math: assumptions, three allocation modes, efficient frontier, backtest, Monte Carlo. |
| `requirements.txt` | Python dependencies. |
| `comparison.md` | Week 3 write-up comparing the heuristic and research modes. |

Upload exactly these three files to the Hugging Face Space: `app.py`, `engine.py`, `requirements.txt`.

## How to run

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Open the local URL Gradio prints (normally http://127.0.0.1:7860). The page builds Client A on load; the two preset buttons swap in Client A or Client B. Every input change rebuilds the outputs.

To print the baseline numbers without the UI:

```bash
python engine.py
```

## Test clients

**Client A:** age 35, Moderate, 30-year horizon, $100,000 initial, $2,000/month, Retirement, income $140,000.

**Client B:** age 68, Moderate, 20-year horizon, $1,500,000 initial, $0/month, Retirement, income $48,000.

Equity share by engine:

| Client | Heuristic (110 − age) | Mean-variance | Research-informed |
|---|---|---|---|
| A | 75.0% | 64.2% | 90.4% |
| B | 42.0% | 64.2% | 60.8% |

Mode C internals: Client A has H/W ≈ 31, so Choi's α̂ saturates at 100% and the blend lands at 90.4%. Client B has H ≈ $765k against W = $1.5M (H/W = 0.51), α* = 41.0%, α̂ = 61.9%, and the blend 0.6 × 60% + 0.4 × 61.9% = 60.8%, inside the 55–65% target band.

Mean-variance is age-blind: it targets 11% volatility for a Moderate client and returns the same 64.2% for both.

## References

Duarte, V., Fonseca, J., Goodman, A. S., & Parker, J. A. (2021). *Simple Allocation Rules and Optimal Portfolio Choice Over the Lifecycle.* NBER Working Paper 29559.
Choi, J. J. (2022). Popular Personal Financial Advice versus the Professors. *Journal of Economic Perspectives*, 36(4).
Choi, J. J., Liu, S., & Liu, Y. (2025). *Practical Finance.* NBER Working Paper 34166.
