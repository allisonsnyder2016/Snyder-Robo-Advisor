---
title: NVIDIA DCF — Teaching App
emoji: 📉
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# NVIDIA DCF — teaching app

**Valuation date: July 29/30, 2026. Market price $190.01 per share, 24.22B shares
outstanding. This is not investment advice.**

An interactive discounted cash flow model built to teach DCF mechanics. You set
every assumption; the app runs the arithmetic and shows what those assumptions
imply. Nothing here is a recommendation, a price target, or the output of a
licensed investment adviser.

## What the app does

Given a base revenue and a ten-year path of growth rates and operating margins,
the app builds a year-by-year forecast, discounts it, and bridges enterprise
value to an intrinsic value per share:

```
FCF   = EBIT x (1 - tax) + D&A - CapEx - dNWC
TV    = FCF_N x (1 + g) / (WACC - g)
EV    = PV(explicit FCF) + PV(TV)
Equity = EV - net debt          (net debt is negative here, so net cash is added)
IV/share = Equity / shares
```

It then shows four things: the year-by-year table and valuation bridge, a
probability-weighted Bear/Base/Bull panel, two sensitivity heatmaps, and a
reverse "what would justify today's price" break-even.

### Repository layout

| File | Contents |
| --- | --- |
| `dcf.py` | All valuation math. Pure functions, pandas + numpy only, no Gradio. Runnable on its own with `python dcf.py`. |
| `app.py` | Gradio UI. Collects inputs, calls `dcf.py`, formats the output. Contains no valuation math. |
| `tests/test_dcf.py` | 27 pytest assertions pinning the math and the case result. |

## Run it locally

```bash
pip install -r requirements.txt && python app.py
```

Then open the local URL Gradio prints. To run the model without the UI:

```bash
python dcf.py
```

To run the tests:

```bash
pytest -q
```

## How to use it

### Main controls (left column)

These are the live "custom" case. Everything on the page recomputes from them.

- **Company & market** — base revenue, shares, net debt, market price.
- **Operating drivers** — forecast horizon (1–15 years), tax rate, gross CapEx
  %, D&A %, and NWC as a percentage of the *change* in revenue.
- **Discount rate** — a radio switches between typing a WACC directly and
  building one from CAPM (`rf + beta x ERP`, blended with after-tax cost of debt
  by the debt weight). The WACC actually used is printed at the top of the
  report.
- **Growth and margin by year** — ten boxes each, one per forecast year. Years
  2–9 are yours to edit; nothing is interpolated. If you set the horizon past
  ten years, the "long-term growth" box fills years 11+.
- **Terminal growth** — must be below WACC. If it is not, a red banner appears,
  the valuation is skipped, and the charts render empty rather than crashing.

Click **Run valuation** to recompute. The report also renders on page load.

### Heatmap modes

A dropdown under the charts switches the sensitivity grid:

- **Mode A — terminal g x year-10 margin** (default). Rows are terminal growth
  from 1.5% to 4.5%; columns are the year-10 operating margin from 25% to 55%.
  Only the last forecast year's margin is replaced; years 1–9 stay as you set
  them.
- **Mode B — WACC x year-1 growth.** Rows are WACC from 9% to 14%; columns are
  year-1 revenue growth from 20% to 80%.

Every cell is a full DCF re-run, not an interpolation. Colour is centred on the
market price, so blue cells are above it and red cells below. The base case is
outlined in solid black and the cell closest to the market price is outlined in
dotted green. Any cell where terminal growth would meet or exceed WACC shows
`n/a` instead of a number.

### Bear / Base / Bull

Each scenario has five controls: probability, year-1 growth, terminal growth,
year-10 operating margin, and WACC. Everything a scenario does not name — base
revenue, tax, CapEx, D&A, NWC, net debt, shares, horizon, and the year 2–9
growth and margin path — is inherited from the main sliders, so the panel tracks
your custom case.

Probabilities must sum to 1.000 ± 0.001 or the weighted value is withheld and an
error is shown. A scenario whose terminal growth is not below its own WACC is
marked `invalid` and dropped from the weighted average.

| Scenario | Probability | Year-1 growth | Terminal g | Year-10 margin | WACC |
| --- | --- | --- | --- | --- | --- |
| Bear | 0.25 | 0.25 | 0.025 | 0.28 | 0.125 |
| Base | 0.50 | 0.60 | 0.030 | 0.36 | 0.110 |
| Bull | 0.25 | 0.80 | 0.035 | 0.48 | 0.100 |

The recommendation is a mechanical rule, printed in the app:

> Buy if weighted IV > price x 1.15; Sell if weighted IV < price x 0.85; Hold
> otherwise.

## Default assumptions

| Input | Default | Source |
| --- | --- | --- |
| Base revenue (FY2026) | 215.938 ($B) | Exhibit 1 |
| Tax rate | 0.17 | Exhibit 2B (16–18% range, midpoint taken) |
| Gross CapEx % of revenue | 0.028 | Exhibit 5 |
| D&A % of revenue | 0.013 | Exhibit 5 |
| NWC % of the change in revenue | 0.12 | **My judgment**, against FY26 actual of 0.193 |
| Net debt | -115.466 ($B, i.e. net cash) | Exhibit 4/6 |
| Shares outstanding | 24.22 (B) | Exhibit 6/7 |
| Market price | 190.01 ($/share) | Exhibit 6/7 |
| Risk-free rate | 0.047 | Exhibit 6 |
| Equity risk premium | 0.0423 | Exhibit 6 |
| Beta | 1.55 | Exhibit 6 — **against a regression beta of 2.21** |
| WACC | 0.11 | My assumption (direct entry; CAPM mode available) |
| Terminal growth | 0.03 | My assumption |
| Growth taper, years 1–10 | 0.60, 0.40, 0.28, 0.20, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04 | **My assumption** |
| Margin taper, years 1–10 | 0.60, 0.58, 0.55, 0.52, 0.48, 0.45, 0.42, 0.40, 0.38, 0.36 | **My assumption** |

Two of these deserve emphasis. The NWC ratio of 0.12 is a judgment call well
below the FY26 actual of 0.193, which flatters free cash flow. The beta of 1.55
is materially below the 2.21 regression beta; using 2.21 would raise the cost of
equity and lower the intrinsic value. The growth and margin tapers are mine and
are the single largest driver of the result.

## Default output

Running the app untouched produces:

| Figure | Value |
| --- | --- |
| Intrinsic value per share | **$131.33** |
| Market price | $190.01 |
| Upside / (downside) | −30.88% |
| Enterprise value | $3,065.4 B |
| Equity value | $3,180.9 B |
| Terminal value as % of EV | 48.10% |
| Bear / Base / Bull IV | $77.31 / $131.33 / $210.04 |
| Probability-weighted IV | **$137.50** |
| Recommendation (rule output) | **SELL** |

The weighted $137.50 sits below the sell threshold of $161.51 (price x 0.85), so
the rule returns SELL. That is arithmetic, not a view.

Terminal value carries 48.10% of enterprise value at these settings — worth
noticing before treating the answer as precise.

## Break-even (section 6.1)

The app also runs the question backwards: what would have to be true to justify
$190.01? Using a steady-state perpetuity:

```
EV           = price x shares + net debt   = $4,486.6 B
required FCF = EV x (WACC - g)             = $358.9 B
required rev = required FCF / FCF margin   = ~$855 B
```

That is roughly **$855B of run-rate revenue at a 42% FCF margin, an 11% WACC and
3% terminal growth** — about 3.96x FY26 revenue of $215.9B, 3.37x TTM of
$253.5B, and 2.17x street FY27 of $393.6B. Whether that is plausible is the
actual question the model is asking you.

## Two terminal value conventions — deliberate

The app uses two different perpetuity formulas, and they are not the same:

| Where | Formula | Note |
| --- | --- | --- |
| Forecast terminal value | `FCF_N x (1 + g) / (WACC - g)` | Gordon growth, with the `(1 + g)` step-up |
| Section 6.1 break-even | `EV = FCF / (WACC - g)`, so `required FCF = EV x (WACC - g)` | No `(1 + g)` in the numerator |

This is intentional and follows the case's own section 6.1 convention. Because
the Gordon form credits a year of growth before discounting, it needs *less*
current FCF to support the same enterprise value: applying it to the break-even
would lower required FCF from $358.9B to $348.5B and required revenue from
$854.6B to roughly $830B. The section 6.1 convention is therefore the more
demanding of the two, by about $25B of revenue.

## Limitations

Read these before drawing any conclusion from the output.

- **No stock-based compensation add-back or adjustment.** SBC is a real cost to
  shareholders and is not modelled separately.
- **No R&D capitalization.** R&D is expensed inside the operating margin rather
  than capitalized and amortized, which affects both invested capital and the
  margin path.
- **No segment model.** Revenue is a single consolidated line. Data centre,
  gaming, networking and automotive are not forecast separately, so mix shift
  cannot be reasoned about.
- **Non-marketable AI stakes are inside the non-operating assets.** Roughly $43B
  of non-marketable AI investments sit within the $115.5B of non-operating
  assets net of debt, and are added to equity value at carrying value. Those
  stakes are correlated with the same AI demand that drives the forecast, so
  they are not an independent cushion: in the scenario where the revenue path
  disappoints, the stakes are likely impaired at the same time.
- **Circular financing is not in the cash flows.** Vendor financing, equity
  stakes in customers, and similar arrangements that can inflate reported demand
  are not modelled.
- **Terminal value dominates.** Just under half of enterprise value at default
  settings comes from the perpetuity, which is the least reliable part of any
  DCF.
- **Single-point tax and working capital ratios.** Both are held flat across the
  entire forecast.

## Not investment advice

This is a teaching tool. It is not investment advice, not a recommendation, and
not a price target. The author is not a licensed investment adviser. The
"BUY / SELL / HOLD" label is the mechanical output of a printed threshold rule
applied to your own assumptions, nothing more. Do your own work.
