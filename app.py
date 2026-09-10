"""Gradio front end for the student robo-advisor.

This module contains NO portfolio math. Every number on screen comes from a
call into engine.py; app.py only validates the widgets, calls the engine,
and turns the results into markdown and Plotly figures.
"""

import os

import numpy as np
import plotly.graph_objects as go
import gradio as gr

import engine

# Optional live market data. Off by default; set ROBO_LIVE_DATA=1 to try a
# yfinance download. If it fails for any reason the engine keeps its
# hardcoded moments and DATA_SOURCE says so. Must run before FRONTIER below.
if os.environ.get("ROBO_LIVE_DATA", "").strip().lower() in ("1", "true", "yes"):
    engine.use_live_data()
DATA_NOTE = f"Market assumptions in use: {engine.DATA_SOURCE}."

# --------------------------------------------------------------------------
# Constants for the UI
# --------------------------------------------------------------------------

DISCLAIMER = (
    "**Educational prototype. Not investment advice.** "
    "Expected returns, vols and correlations come from `engine.py` "
    "(hardcoded long-run assumptions unless a yfinance download was "
    "requested and succeeded); the 2007–2024 backtest uses approximate "
    "historical returns."
)

RISK_CHOICES = engine.RISK_LEVELS
GOAL_CHOICES = ["Retirement", "Home purchase", "Education", "General wealth"]

ENGINE_HEURISTIC = "Heuristic (110 − age)"
ENGINE_MVO = "Mean-variance"
ENGINE_RESEARCH = "Research-informed (Duarte + Choi)"
# The radio only exposes the methods engine.public_methods() allows. While
# engine.RESEARCH_MODE_ENABLED is False that is heuristic + mean-variance;
# the research strings stay defined so turning Mode C back on is one flag.
_ALL_ENGINE_METHOD = {ENGINE_HEURISTIC: "heuristic", ENGINE_MVO: "mean_variance",
                      ENGINE_RESEARCH: "research"}
ENGINE_METHOD = {label: m for label, m in _ALL_ENGINE_METHOD.items()
                 if m in engine.public_methods()}
ENGINE_CHOICES = list(ENGINE_METHOD)
ENGINE_LABEL = {"heuristic": "Heuristic", "mean_variance": "Mean-variance",
                "research": "Research-informed"}
COMPARE_TAB_TITLE = ("Heuristic vs MVO vs Research" if engine.RESEARCH_MODE_ENABLED
                     else "Heuristic vs MVO")

# Preset order matches INPUT_ORDER below:
# age, risk, horizon, initial, monthly, goal, engine, income, goal_target
PRESET_A = [35, "Moderate", 30, 100_000, 2_000, "Retirement", ENGINE_HEURISTIC, 140_000, 2_500_000]
PRESET_B = [68, "Moderate", 20, 1_500_000, 0, "Retirement", ENGINE_HEURISTIC, 48_000, 1_800_000]

AGE_MIN, AGE_MAX = 18, 80
HORIZON_MIN, HORIZON_MAX = 1, 40
INITIAL_MIN = 1_000

BENCHMARK_6040 = {"VTI": 0.60, "BND": 0.40}

# The long-only frontier does not depend on any client input, so compute it
# once at import time (30 SLSQP solves) instead of on every rebuild.
FRONTIER = engine.efficient_frontier(n_points=30)

N_OUTPUTS = 7  # metrics md, pie, scatter, wealth, backtest, drawdown, table md


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate(age, risk, horizon, initial, monthly, goal, engine_choice,
              income, goal_target):
    """Return a list of human-readable problems (empty list = all good)."""
    problems = []
    if age is None or not (AGE_MIN <= age <= AGE_MAX):
        problems.append(f"Age must be between {AGE_MIN} and {AGE_MAX}.")
    if risk not in RISK_CHOICES:
        problems.append(f"Risk must be one of {', '.join(RISK_CHOICES)}.")
    if horizon is None or not (HORIZON_MIN <= horizon <= HORIZON_MAX):
        problems.append(f"Horizon must be between {HORIZON_MIN} and {HORIZON_MAX} years.")
    if initial is None or initial < INITIAL_MIN:
        problems.append(f"Initial investment must be at least ${INITIAL_MIN:,.0f}.")
    if monthly is None or monthly < 0:
        problems.append("Monthly contribution cannot be negative.")
    if goal not in GOAL_CHOICES:
        problems.append(f"Goal must be one of {', '.join(GOAL_CHOICES)}.")
    if engine_choice not in ENGINE_CHOICES:
        problems.append("Pick an allocation engine.")
    if income is None or income < 0:
        problems.append("Annual income cannot be negative.")
    if goal_target is None or goal_target <= 0:
        problems.append("Goal target must be positive.")
    return problems


def _error_outputs(problems):
    msg = "### Fix these inputs\n" + "\n".join(f"- {p}" for p in problems)
    return [msg] + [go.Figure() for _ in range(N_OUTPUTS - 2)] + [""]


# --------------------------------------------------------------------------
# Figure builders (pure formatting of engine results)
# --------------------------------------------------------------------------


def _pie(weights):
    wd = engine.weights_to_dict(weights)
    labels = [f"{t} · {engine.ASSET_NAMES[t]}" for t in engine.TICKERS]
    fig = go.Figure(go.Pie(
        labels=labels, values=[wd[t] for t in engine.TICKERS],
        hole=0.4, sort=False, textinfo="label+percent",
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Recommended weights", margin=dict(l=20, r=20, t=50, b=20))
    return fig


def _scatter(weights, stats, label, method):
    fig = go.Figure()
    # The efficient frontier is a mean-variance concept; only draw it when
    # the mean-variance engine produced the recommendation.
    if method == "mean_variance":
        fig.add_trace(go.Scatter(
            x=FRONTIER["vols"], y=FRONTIER["returns"], mode="lines",
            name="Long-only efficient frontier", line=dict(width=2),
            hovertemplate="vol %{x:.1%}<br>E[r] %{y:.2%}<extra>Frontier</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=engine.VOLATILITIES, y=engine.EXPECTED_RETURNS, mode="markers+text",
        text=engine.TICKERS, textposition="top center", name="Assets",
        marker=dict(size=10),
        hovertemplate="%{text}<br>vol %{x:.1%}<br>E[r] %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[stats["volatility"]], y=[stats["expected_return"]], mode="markers",
        name=f"Recommended ({label})",
        marker=dict(symbol="star", size=18, line=dict(width=1)),
        hovertemplate=(f"Recommended<br>vol %{{x:.1%}}<br>E[r] %{{y:.2%}}"
                       f"<br>Sharpe {stats['sharpe']:.2f}<extra></extra>"),
    ))
    fig.update_layout(
        title="Risk vs. return (annual, long-run assumptions)",
        xaxis=dict(title="Volatility", tickformat=".0%"),
        yaxis=dict(title="Expected return", tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _wealth(mc, initial, monthly, horizon, goal_target):
    months = mc["months"]
    x = np.arange(months + 1) / 12.0
    contributed = initial + monthly * np.arange(months + 1)
    pct = mc["path_percentiles"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pct[75], name="P75", mode="lines",
                             line=dict(width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=pct[25], name="P25", mode="lines",
                             line=dict(width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(100,100,200,0.15)"))
    fig.add_trace(go.Scatter(x=x, y=mc["path_mean"], name="Mean", mode="lines",
                             line=dict(width=3)))
    fig.add_trace(go.Scatter(x=x, y=contributed, name="Contributed capital",
                             mode="lines", line=dict(width=1, dash="dashdot")))
    fig.add_hline(y=goal_target, line=dict(dash="dash", width=2),
                  annotation_text=f"Goal ${goal_target:,.0f}",
                  annotation_position="top left")
    fig.update_layout(
        title=f"Monte Carlo wealth projection ({mc['n_paths']} paths, {horizon}y)",
        xaxis_title="Years", yaxis=dict(title="Wealth ($)", tickformat="$,.0f"),
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _backtest(bt, bt_6040, label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["years"], y=bt["path"], mode="lines",
                             name=f"Recommended ({label})", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=bt_6040["years"], y=bt_6040["path"], mode="lines",
                             name="60/40 VTI/BND", line=dict(width=2, dash="dash")))
    fig.update_layout(
        title=(f"Backtest 2007–2024, growth of $1 (annual rebalance) · "
               f"CAGR {bt['cagr']:.2%} vs 60/40 {bt_6040['cagr']:.2%}"),
        xaxis_title="Year", yaxis=dict(title="Growth of $1", tickformat="$.2f"),
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _drawdown(bt):
    path = np.asarray(bt["path"])
    dd = path / np.maximum.accumulate(path) - 1.0
    years = bt["years"]
    trough = int(np.argmin(dd))
    worst_year, worst_ret = bt["worst_year"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=dd, mode="lines", fill="tozeroy",
                             name="Drawdown", line=dict(width=2),
                             hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.add_annotation(
        x=years[trough], y=dd[trough],
        text=f"Max drawdown {dd[trough]:.1%} ({years[trough]})",
        showarrow=True, arrowhead=2, ax=40, ay=40,
    )
    fig.update_layout(
        title=f"Drawdown of recommended mix · worst year {worst_year}: {worst_ret:.1%}",
        xaxis_title="Year", yaxis=dict(title="Drawdown from peak", tickformat=".0%"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _metrics_md(rec, label, goal, goal_target, income, monthly, p_goal):
    s, bt = rec.stats, rec.backtest
    contrib_pct = (12 * monthly / income) if income else 0.0
    wd = rec.weights_dict
    weights_line = ", ".join(f"{t} {w:.1%}" for t, w in wd.items() if w > 0.0005)
    return (
        f"### Recommended portfolio · {label}\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Expected return | {s['expected_return']:.2%} |\n"
        f"| Expected volatility | {s['volatility']:.2%} |\n"
        f"| Sharpe (rf {engine.RISK_FREE_RATE:.0%}) | {s['sharpe']:.2f} |\n"
        f"| Max historical drawdown (2007–2024) | {bt['max_drawdown']:.1%} |\n"
        f"| P(wealth ≥ ${goal_target:,.0f} goal) | {p_goal:.0%} |\n"
        f"| Equity share | {s['equity_share']:.1%} |\n"
        f"| Diversification benefit | {s['diversification_benefit']:.2%} vol saved |\n\n"
        f"**Weights:** {weights_line}\n\n"
        f"Goal: {goal}. Contributions are {contrib_pct:.0%} of annual income."
    )


def _compare_md(client, chosen_method):
    cols = [
        ("Heuristic (110 − age)", "heuristic",
         engine.heuristic_allocation(client.age, client.risk)),
        ("Mean-variance", "mean_variance",
         engine.mean_variance_allocation(client.risk)),
    ]
    if engine.RESEARCH_MODE_ENABLED:
        cols.append(("Research-informed", "research",
                     engine.research_allocation(client)))
    stats = [engine.portfolio_stats(w) for _, _, w in cols]
    header = "| Metric | " + " | ".join(
        f"**{name}** ◀" if method == chosen_method else name
        for name, method, _ in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [
        ("Equity %", [f"{st['equity_share']:.1%}" for st in stats]),
        ("E[r]", [f"{st['expected_return']:.2%}" for st in stats]),
        ("E[vol]", [f"{st['volatility']:.2%}" for st in stats]),
        ("Sharpe", [f"{st['sharpe']:.2f}" for st in stats]),
    ]
    table = "\n".join([header, sep] + [f"| {m} | " + " | ".join(v) + " |" for m, v in rows])
    text = (
        f"### {COMPARE_TAB_TITLE}\n" + table +
        f"\n\nHeuristic uses age only. Mean-variance targets {engine.TARGET_VOL[client.risk]:.0%} "
        f"volatility for {client.risk} and ignores age."
    )
    if engine.RESEARCH_MODE_ENABLED:
        text += (" Research-informed blends the Duarte lifecycle glide with a Choi "
                 "human-capital tilt.\n\n" + _research_narrative(client))
    return text


def _research_narrative(client):
    d = engine.research_equity_share(client)
    diff = d["equity"] - d["popular_equity"]
    direction = "more" if diff >= 0 else "less"
    income_kind = "pension / Social Security-like income" if d["retired"] else \
        f"remaining wages plus a pension / Social Security-like income after {engine.RETIREMENT_AGE}"
    if d["income_inferred"]:
        income_kind += (f" (no income entered, so a ${d['income']:,.0f}/yr stand-in is used; "
                        "a retiree's human capital is never set to zero)")
    tilt_txt = f" and a {d['tilt']:+.0%} {client.risk} tilt" if d["tilt"] else ""
    return (
        "#### Where Mode C disagrees with 110 − age, and why\n"
        f"For this client Mode C holds **{d['equity']:.1%}** equity versus "
        f"**{d['popular_equity']:.1%}** under 110 − age, about {abs(diff):.0%} {direction}.\n\n"
        f"- **Lifecycle glide (Duarte et al. 2021):** {d['duarte']:.0%} at age {client.age}. "
        "The optimal equity share of financial wealth is hump-shaped, near 80% around 45, "
        f"and settles at a flat ~60% from {engine.RETIREMENT_AGE} on. Typical target-date funds fall to 30–40% "
        "by the late 60s, which the paper estimates costs 2–3% of consumption.\n"
        f"- **Human capital is an implicit bond (Choi 2022; Choi, Liu & Liu 2025):** "
        f"the present value of {income_kind} is H ≈ ${d['H']:,.0f} against financial wealth "
        f"W = ${d['W']:,.0f}, so H/W = {d['H_over_W']:.2f}. With γ = {d['gamma']:.0f}, "
        f"α* = {d['alpha_star']:.1%} of total wealth becomes α̂ = {d['alpha_hat']:.0%} of "
        "financial wealth, because the bond-like income already supplies the safe part.\n"
        "- **TDFs cut financial equity twice:** human capital shrinks with age on its own, "
        "and an age-only rule then cuts the financial-equity share as well. Mode C uses "
        f"{engine.RESEARCH_BLEND_DUARTE:.0%} Duarte + {1 - engine.RESEARCH_BLEND_DUARTE:.0%} "
        f"Choi{tilt_txt}, giving {d['blend'] + d['tilt']:.1%} before the "
        f"{engine.EQUITY_FLOOR:.0%}–{engine.EQUITY_CAP:.0%} clip. 110 − age stays as the "
        "popular-advice foil.\n"
    )


# --------------------------------------------------------------------------
# The single build function every event calls
# --------------------------------------------------------------------------


def build(age, risk, horizon, initial, monthly, goal, engine_choice,
          income, goal_target):
    problems = _validate(age, risk, horizon, initial, monthly, goal,
                         engine_choice, income, goal_target)
    if problems:
        return _error_outputs(problems)

    age, horizon = int(age), int(horizon)
    initial, monthly = float(initial), float(monthly)
    income, goal_target = float(income), float(goal_target)
    method = ENGINE_METHOD[engine_choice]
    label = ENGINE_LABEL[method]

    try:
        client = engine.ClientProfile(
            age=age, risk=risk, horizon_years=horizon, initial=initial,
            monthly_contribution=monthly, goal=goal, annual_income=income,
        )
        rec = engine.recommend(client, method)
        bt_1 = engine.backtest(rec.weights, initial=1.0)
        bt_6040 = engine.backtest(BENCHMARK_6040, initial=1.0)
        p_goal = float(np.mean(rec.monte_carlo["terminal"] >= goal_target))
    except (ValueError, RuntimeError) as exc:
        return _error_outputs([str(exc)])

    return [
        _metrics_md(rec, label, goal, goal_target, income, monthly, p_goal),
        _pie(rec.weights),
        _scatter(rec.weights, rec.stats, label, method),
        _wealth(rec.monte_carlo, initial, monthly, horizon, goal_target),
        _backtest(bt_1, bt_6040, label),
        _drawdown(bt_1),
        _compare_md(client, method),
    ]


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

with gr.Blocks(title="Student Robo-Advisor") as demo:
    gr.Markdown("# Student Robo-Advisor")
    gr.Markdown(DISCLAIMER)
    gr.Markdown(DATA_NOTE)

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### Client profile")
            age = gr.Slider(AGE_MIN, AGE_MAX, value=35, step=1, label="Age")
            risk = gr.Radio(RISK_CHOICES, value="Moderate", label="Risk tolerance")
            horizon = gr.Slider(HORIZON_MIN, HORIZON_MAX, value=30, step=1,
                                label="Horizon (years)")
            initial = gr.Number(value=100_000, label="Initial investment ($)",
                                minimum=0, precision=0)
            monthly = gr.Number(value=2_000, label="Monthly contribution ($)",
                                minimum=0, precision=0)
            goal = gr.Dropdown(GOAL_CHOICES, value="Retirement", label="Goal")
            engine_choice = gr.Radio(ENGINE_CHOICES, value=ENGINE_HEURISTIC,
                                     label="Allocation engine")
            income = gr.Number(value=140_000, label="Annual income ($)",
                               minimum=0, precision=0)
            goal_target = gr.Number(value=2_500_000, label="Goal target ($)",
                                    minimum=0, precision=0)
            run = gr.Button("Build portfolio", variant="primary")
            with gr.Row():
                load_a = gr.Button("Load client A")
                load_b = gr.Button("Load client B")

        with gr.Column(scale=2):
            metrics_md = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("Weights"):
                    pie_plot = gr.Plot()
                with gr.Tab("Risk / return"):
                    scatter_plot = gr.Plot()
                with gr.Tab("Wealth projection"):
                    wealth_plot = gr.Plot()
                with gr.Tab("Backtest"):
                    backtest_plot = gr.Plot()
                with gr.Tab("Drawdown"):
                    drawdown_plot = gr.Plot()
                with gr.Tab(COMPARE_TAB_TITLE):
                    compare_md = gr.Markdown()

    INPUT_ORDER = [age, risk, horizon, initial, monthly, goal,
                   engine_choice, income, goal_target]
    OUTPUTS = [metrics_md, pie_plot, scatter_plot, wealth_plot,
               backtest_plot, drawdown_plot, compare_md]

    run.click(build, INPUT_ORDER, OUTPUTS)
    for comp in INPUT_ORDER:
        comp.change(build, INPUT_ORDER, OUTPUTS)
    demo.load(build, INPUT_ORDER, OUTPUTS)

    # Presets only set the inputs; each input's .change then rebuilds.
    load_a.click(lambda: PRESET_A, None, INPUT_ORDER)
    load_b.click(lambda: PRESET_B, None, INPUT_ORDER)


if __name__ == "__main__":
    demo.launch()
