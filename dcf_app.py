"""Gradio front end for the NVIDIA DCF.

This module contains NO valuation math. Every number on screen comes from a
call into dcf.py; app.py only assembles the per-year lists from the widgets,
calls dcf, and formats the result as text.
"""

import gradio as gr
import plotly.graph_objects as go

import dcf

VALUATION_DATE = "July 29/30, 2026"

# Default control values. Single source of truth for the widgets, so the
# defaults on screen and the defaults used in any headless check agree.
DEFAULTS = {
    "base_revenue": 215.938,
    "shares": 24.22,
    "net_debt": -115.466,
    "current_price": 190.01,
    "years": 10,
    "tax_rate": 0.17,
    "capex_pct": 0.028,
    "da_pct": 0.013,
    "nwc_pct": 0.12,
    "wacc_mode": "Direct WACC",
    "direct_wacc": 0.11,
    "rf": 0.047,
    "erp": 0.0423,
    "beta": 1.55,
    "after_tax_kd": 0.04,
    "debt_weight": 0.0,
    "terminal_g": 0.03,
    "growth": [0.60, 0.40, 0.28, 0.20, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04],
    "long_term_growth": 0.04,
    "margins": [0.60, 0.58, 0.55, 0.52, 0.48, 0.45, 0.42, 0.40, 0.38, 0.36],
}

# Revenue comparators for the section 6.1 readout ($B, case facts)
REV_FY26 = 215.9
REV_TTM = 253.5
REV_STREET_FY27 = 393.6

# Section 6.1 steady-state FCF margin (dcf.py's own default)
STEADY_FCF_MARGIN = 0.42


# ---------------------------------------------------------------------------
# widget values -> per-year lists (assembly only, no math)
# ---------------------------------------------------------------------------


def build_per_year(front, back, tail, years):
    """Years 1-5 from `front`, years 6-10 from `back`, years 11+ from `tail`.

    Truncates to `years`. No interpolation: the ten numbers on screen are the
    ten numbers handed to dcf.project_financials.
    """
    years = int(years)
    values = list(front) + list(back)
    if years <= len(values):
        return values[:years]
    if tail is None:
        return values
    return values + [tail] * (years - len(values))


def resolve_wacc(wacc_mode, direct_wacc, rf, erp, beta, after_tax_kd, debt_weight):
    """Returns (wacc, label). Both branches go through dcf.compute_wacc."""
    if wacc_mode == "Direct WACC":
        wacc = dcf.compute_wacc(0, 0, 0, 0, 0, override=direct_wacc)
        return wacc, f"{wacc:.4%}  (direct entry)"
    wacc = dcf.compute_wacc(rf, erp, beta, after_tax_kd, debt_weight)
    # Ke comes back out of dcf.compute_wacc at zero debt weight rather than
    # being re-derived here, so app.py holds no copy of the CAPM formula.
    cost_of_equity = dcf.compute_wacc(rf, erp, beta, 0.0, 0.0)
    label = (
        f"{wacc:.4%}  (CAPM: rf {rf:.2%} + beta {beta:.2f} x ERP {erp:.2%} "
        f"= Ke {cost_of_equity:.4%}; debt weight {debt_weight:.1%} "
        f"at after-tax Kd {after_tax_kd:.2%})"
    )
    return wacc, label


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _banner(message):
    return (
        '<div style="border-left:6px solid #c0392b;background:#fdeceb;'
        'padding:12px 16px;border-radius:4px;">'
        '<strong style="color:#c0392b;">VALUATION SKIPPED</strong><br>'
        f'<span style="color:#7b241c;">{message}</span></div>'
    )


def money_b(value):
    """$ billions, one decimal: $3,065.4 B / -$115.5 B"""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.1f} B"


def money_ps(value):
    """$ per share, two decimals: $131.33 / -$4.20"""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _kpi(label, value, width=20):
    return f"{label:<34}{value:>{width}}"


def _table(proj):
    cols = ["revenue", "op_margin", "ebit", "nopat", "da", "capex",
            "delta_nwc", "fcf", "pv_fcf"]
    table = proj[cols].rename(
        columns={"revenue": "rev", "op_margin": "margin", "delta_nwc": "dnwc"}
    )
    formatters = {c: (lambda v: f"{v:,.1f}")
                  for c in ["rev", "ebit", "nopat", "da", "capex", "dnwc",
                            "fcf", "pv_fcf"]}
    formatters["margin"] = lambda v: f"{v:.1%}"
    return table.to_string(formatters=formatters)


# ---------------------------------------------------------------------------
# the single valuation path: text, charts and every heatmap cell go through
# this one function, so there is exactly one copy of the dcf call sequence
# ---------------------------------------------------------------------------


def value_scenario(base_revenue, growth_rates, op_margins, tax_rate, capex_pct,
                   nwc_pct, da_pct, years, wacc, terminal_g, net_debt, shares):
    """Run one full DCF through dcf.py. Returns (proj, tv, disc, bridge).

    Raises ValueError (from dcf.gordon_tv) when terminal growth >= WACC.
    """
    proj = dcf.project_financials(
        base_revenue=base_revenue,
        growth_rates=growth_rates,
        op_margins=op_margins,
        tax_rate=tax_rate,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        da_pct=da_pct,
        years=years,
    )
    proj["fcf"] = dcf.compute_fcf(proj)
    tv = dcf.gordon_tv(float(proj["fcf"].iloc[-1]), terminal_g, wacc)
    disc = dcf.discount_cash_flows(proj["fcf"], tv, wacc)
    proj["pv_fcf"] = disc["pv_fcf_by_year"]
    bridge = dcf.equity_value_per_share(disc["enterprise_value"], net_debt, shares)
    return proj, tv, disc, bridge


# ---------------------------------------------------------------------------
# Bear / Base / Bull
# ---------------------------------------------------------------------------

# (name, probability, year-1 growth, terminal g, year-10 margin, WACC)
SCENARIO_DEFAULTS = [
    ("Bear", dict(prob=0.25, y1_g=0.25, terminal_g=0.025, y10_m=0.28, wacc=0.125)),
    ("Base", dict(prob=0.50, y1_g=0.60, terminal_g=0.030, y10_m=0.36, wacc=0.110)),
    ("Bull", dict(prob=0.25, y1_g=0.80, terminal_g=0.035, y10_m=0.48, wacc=0.100)),
]

PROB_TOLERANCE = 0.001
BUY_MULTIPLE = 1.15
SELL_MULTIPLE = 0.85

RECOMMENDATION_RULE = (
    f"Buy if weighted IV > price x {BUY_MULTIPLE:.2f}; "
    f"Sell if weighted IV < price x {SELL_MULTIPLE:.2f}; Hold otherwise."
)


def parse_scenario_values(scenario_values):
    """Flat widget list -> [(name, params), ...]. Falls back to the defaults."""
    if len(scenario_values) != 5 * len(SCENARIO_DEFAULTS):
        return [(name, dict(params)) for name, params in SCENARIO_DEFAULTS]
    out = []
    for idx, (name, _) in enumerate(SCENARIO_DEFAULTS):
        prob, y1_g, terminal_g, y10_m, wacc = scenario_values[idx * 5:idx * 5 + 5]
        out.append((name, dict(prob=prob, y1_g=y1_g, terminal_g=terminal_g,
                               y10_m=y10_m, wacc=wacc)))
    return out


def scenario_iv(base, params):
    """One scenario's IV/share, through the same value_scenario as everything.

    Starts from the current main-slider growth and margin lists, then overrides
    only year-1 growth, the last used year's margin, WACC and terminal g.
    Year-10 GROWTH is deliberately left on the main-slider taper -- see the
    note in the panel: overwriting it with terminal g moves Base off 131.3331.
    """
    kw = dict(base)
    growth = list(kw["growth_rates"])
    growth[0] = params["y1_g"]
    margins = list(kw["op_margins"])
    margins[-1] = params["y10_m"]
    kw["growth_rates"] = growth
    kw["op_margins"] = margins
    kw["wacc"] = params["wacc"]
    kw["terminal_g"] = params["terminal_g"]
    _, _, _, bridge = value_scenario(**kw)
    return bridge["value_per_share"]


def evaluate_scenarios(base, scenarios):
    """[(name, params, iv or None)], probability error string or None."""
    rows = []
    for name, params in scenarios:
        try:
            rows.append((name, params, scenario_iv(base, params)))
        except ValueError:
            rows.append((name, params, None))   # g >= WACC: excluded from weight

    total = sum(params["prob"] for _, params, _ in rows)
    error = None
    if abs(total - 1.0) > PROB_TOLERANCE:
        error = (f"Probabilities sum to {total:.4f}, not 1.000 "
                 f"(+/- {PROB_TOLERANCE}). Weighted IV not computed.")
    return rows, error


def recommendation(weighted_iv, current_price):
    if weighted_iv > current_price * BUY_MULTIPLE:
        return "BUY"
    if weighted_iv < current_price * SELL_MULTIPLE:
        return "SELL"
    return "HOLD"


def scenario_panel(base, scenarios, current_price):
    rows, error = evaluate_scenarios(base, scenarios)

    lines = ["### Bear / Base / Bull", "```"]
    lines.append(f"{'scenario':<10}{'prob':>8}{'Y1 growth':>12}{'term g':>9}"
                 f"{'Y10 margin':>12}{'WACC':>9}{'IV/share':>12}")
    for name, params, iv in rows:
        shown = money_ps(iv) if iv is not None else "invalid"
        lines.append(
            f"{name:<10}{params['prob']:>8.1%}{params['y1_g']:>12.1%}"
            f"{params['terminal_g']:>9.2%}{params['y10_m']:>12.1%}"
            f"{params['wacc']:>9.2%}{shown:>12}")
    lines.append("```")

    valid = [(name, params, iv) for name, params, iv in rows if iv is not None]
    invalid = [name for name, _, iv in rows if iv is None]
    if invalid:
        lines.append(f"*{', '.join(invalid)} invalid (terminal g >= WACC); "
                     "excluded from the weighted average.*")

    if error:
        lines.append("")
        lines.append(f'<div style="border-left:6px solid #c0392b;'
                     f'background:#fdeceb;padding:10px 14px;border-radius:4px;">'
                     f'<strong style="color:#c0392b;">PROBABILITY ERROR</strong>'
                     f'<br><span style="color:#7b241c;">{error}</span></div>')
    elif valid:
        weighted = sum(params["prob"] * iv for _, params, iv in valid)
        ivs = {name: iv for name, _, iv in valid}
        lines.append("```")
        lines.append(_kpi("Weighted IV / share", money_ps(weighted) + " per share"))
        lines.append(_kpi("Market price", money_ps(current_price) + " per share"))
        if "Bear" in ivs and "Bull" in ivs:
            lines.append(_kpi("Bear-Bull range",
                              f"{money_ps(ivs['Bear'])} to {money_ps(ivs['Bull'])}"))
        lines.append(_kpi("Buy above", money_ps(current_price * BUY_MULTIPLE)))
        lines.append(_kpi("Sell below", money_ps(current_price * SELL_MULTIPLE)))
        lines.append(_kpi("RECOMMENDATION",
                          recommendation(weighted, current_price)))
        lines.append("```")
        lines.append(f"**Rule:** {RECOMMENDATION_RULE} "
                     "Mechanical output of the rule above, not advice.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------

MODE_A = "A - terminal g x Y10 margin"
MODE_B = "B - WACC x Y1 growth"

SENS_G = [0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045]
SENS_MARGIN = [0.25, 0.30, 0.36, 0.40, 0.45, 0.50, 0.55]
SENS_WACC = [0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
SENS_G1 = [0.20, 0.35, 0.50, 0.60, 0.70, 0.80]

PLOT_MARGIN = dict(l=70, r=30, t=70, b=90)


def empty_figure(title, message="Valuation skipped - terminal growth must be below WACC"):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        margin=PLOT_MARGIN,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[dict(text=message, xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(color="#c0392b", size=13))],
    )
    return fig


def figure_projection(proj):
    """Revenue and FCF by forecast year, straight off the projection frame."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(proj.index), y=proj["revenue"], mode="lines+markers",
        name="Revenue", hovertemplate="Year %{x}: $%{y:,.1f}B<extra>Revenue</extra>"))
    fig.add_trace(go.Scatter(
        x=list(proj.index), y=proj["fcf"], mode="lines+markers",
        name="Free cash flow",
        hovertemplate="Year %{x}: $%{y:,.1f}B<extra>FCF</extra>"))
    fig.update_layout(
        title="Revenue and FCF ($B)",
        xaxis_title="Forecast year", yaxis_title="$B",
        hovermode="x unified", margin=PLOT_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(dtick=1)
    return fig


def figure_waterfall(disc, bridge, shares):
    """PV(FCF) + PV(TV) = EV, then the non-operating step, = equity value."""
    non_op_step = -bridge["net_debt"]          # equity = EV - net_debt
    ev = disc["enterprise_value"]
    equity = bridge["equity_value"]
    iv = bridge["value_per_share"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "total", "relative", "total"],
        x=["PV of explicit FCF", "PV of terminal value", "Enterprise value",
           "Non-operating assets net of debt", "Equity value"],
        y=[disc["pv_fcf"], disc["pv_tv"], 0.0, non_op_step, 0.0],
        text=[f"{disc['pv_fcf']:,.1f}", f"{disc['pv_tv']:,.1f}", f"{ev:,.1f}",
              f"{non_op_step:+,.1f}", f"{equity:,.1f}"],
        textposition="outside",
        connector={"line": {"color": "#999"}},
        increasing={"marker": {"color": "#2e86c1"}},
        decreasing={"marker": {"color": "#c0392b"}},
        totals={"marker": {"color": "#566573"}},
    ))
    direction = "adds to" if bridge["net_cash"] else "subtracts from"
    fig.update_layout(
        title="Valuation bridge ($B)",
        yaxis_title="$B", margin=PLOT_MARGIN, showlegend=False,
        annotations=[dict(
            text=(f"<b>Intrinsic value / share: {money_ps(iv)}</b><br>"
                  f"Equity {money_b(equity)} / {shares:,.2f}B shares  |  "
                  f"non-operating assets net of debt {direction} equity "
                  f"(${non_op_step:+,.1f}B)"),
            xref="paper", yref="paper", x=0.5, y=-0.28,
            showarrow=False, font=dict(size=12), align="center")],
    )
    return fig


def _sensitivity_axes(mode, base):
    """(rows, cols, row_label, col_label, variant_fn, base_row, base_col)."""
    if mode == MODE_B:
        def variant(wacc_value, growth_value):
            kw = dict(base)
            kw["wacc"] = wacc_value
            growth = list(kw["growth_rates"])
            growth[0] = growth_value
            kw["growth_rates"] = growth
            return kw
        return (SENS_WACC, SENS_G1, "WACC", "Year-1 revenue growth", variant,
                base["wacc"], base["growth_rates"][0])

    def variant(g_value, margin_value):
        kw = dict(base)
        kw["terminal_g"] = g_value
        # replace only the last used year's margin; dcf.project_financials
        # repeats that last value for any pad years beyond the ten boxes
        margins = list(kw["op_margins"])
        margins[-1] = margin_value
        kw["op_margins"] = margins
        return kw
    return (SENS_G, SENS_MARGIN, "Terminal growth g", "Year-10 operating margin",
            variant, base["terminal_g"], base["op_margins"][-1])


def figure_sensitivity(mode, base, current_price):
    """IV/share across a two-way grid. Every cell re-runs value_scenario."""
    rows, cols, row_label, col_label, variant, base_row, base_col = \
        _sensitivity_axes(mode, base)

    z, text = [], []
    best = None          # (abs gap to price, i, j)
    for i, row_value in enumerate(rows):
        z_row, t_row = [], []
        for j, col_value in enumerate(cols):
            try:
                _, _, _, bridge = value_scenario(**variant(row_value, col_value))
                iv = bridge["value_per_share"]
                z_row.append(iv)
                t_row.append(f"{iv:,.2f}")
                gap = abs(iv - current_price)
                if best is None or gap < best[0]:
                    best = (gap, i, j)
            except ValueError:
                z_row.append(None)      # g >= WACC: gap cell, no crash
                t_row.append("n/a")
        z.append(z_row)
        text.append(t_row)

    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(len(cols))), y=list(range(len(rows))),
        text=text, texttemplate="%{text}", hoverongaps=False,
        colorscale="RdBu", zmid=current_price,
        colorbar=dict(title="IV/share ($)"),
        hovertemplate=(f"{row_label}: %{{customdata[0]}}<br>"
                       f"{col_label}: %{{customdata[1]}}<br>"
                       "IV/share: $%{z:,.2f}<extra></extra>"),
        customdata=[[[f"{r:.3f}", f"{c:.3f}"] for c in cols] for r in rows],
    ))

    shapes = []
    def outline(i, j, color, dash):
        shapes.append(dict(type="rect", x0=j - 0.5, x1=j + 0.5,
                           y0=i - 0.5, y1=i + 0.5, xref="x", yref="y",
                           line=dict(color=color, width=3, dash=dash)))

    def locate(value, axis):
        for k, v in enumerate(axis):
            if abs(v - value) < 1e-9:
                return k
        return None

    bi, bj = locate(base_row, rows), locate(base_col, cols)
    if bi is not None and bj is not None:
        outline(bi, bj, "#111111", "solid")
    if best is not None:
        outline(best[1], best[2], "#1e8449", "dot")

    fig.update_layout(
        title=(f"IV/share sensitivity - {col_label} vs {row_label}<br>"
               f"<sub>solid outline = base case  |  dotted outline = "
               f"closest to market ${current_price:,.2f}  |  "
               f"colour centred on ${current_price:,.2f}</sub>"),
        margin=dict(l=90, r=30, t=90, b=70), shapes=shapes,
        xaxis=dict(title=col_label, tickmode="array",
                   tickvals=list(range(len(cols))),
                   ticktext=[f"{c:.1%}" for c in cols]),
        yaxis=dict(title=row_label, tickmode="array",
                   tickvals=list(range(len(rows))),
                   ticktext=[f"{r:.2%}" for r in rows]),
    )
    return fig


# ---------------------------------------------------------------------------
# the one callback
# ---------------------------------------------------------------------------


def run_valuation(
    base_revenue, shares, net_debt, current_price, years,
    tax_rate, capex_pct, da_pct, nwc_pct,
    wacc_mode, direct_wacc, rf, erp, beta, after_tax_kd, debt_weight,
    terminal_g,
    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, long_term_growth,
    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10,
    heatmap_mode=MODE_A,
    *scenario_values,
):
    years = int(years)
    growth_rates = build_per_year(
        [g1, g2, g3, g4, g5], [g6, g7, g8, g9, g10], long_term_growth, years
    )
    op_margins = build_per_year(
        [m1, m2, m3, m4, m5], [m6, m7, m8, m9, m10], None, years
    )

    try:
        wacc, wacc_label = resolve_wacc(
            wacc_mode, direct_wacc, rf, erp, beta, after_tax_kd, debt_weight
        )

        # one scenario dict drives the text, the charts and every heatmap cell
        base_scenario = dict(
            base_revenue=base_revenue,
            growth_rates=growth_rates,
            op_margins=op_margins,
            tax_rate=tax_rate,
            capex_pct=capex_pct,
            nwc_pct=nwc_pct,
            da_pct=da_pct,
            years=years,
            wacc=wacc,
            terminal_g=terminal_g,
            net_debt=net_debt,
            shares=shares,
        )
        # dcf.gordon_tv is the single place the g < WACC rule lives; a breach
        # raises inside value_scenario and lands in the red banner below.
        proj, tv, disc, bridge = value_scenario(**base_scenario)

        steady = dcf.implied_break_even_revenue(
            current_price=current_price,
            shares=shares,
            net_debt=net_debt,
            wacc=wacc,
            g=terminal_g,
            target_op_margin=op_margins[-1],
            tax=tax_rate,
            capex_pct=capex_pct,
            nwc_pct=nwc_pct,
            years=years,
            da_pct=da_pct,
            case_steady_mode=True,
            fcf_margin=STEADY_FCF_MARGIN,
        )
    except ValueError as exc:
        return (
            _banner(str(exc)),
            "",
            empty_figure("Revenue and FCF ($B)"),
            empty_figure("Valuation bridge ($B)"),
            empty_figure("IV/share sensitivity"),
        )

    iv = bridge["value_per_share"]
    upside = iv / current_price - 1.0
    capex_note = (
        "CapEx is GROSS capex (D&A > 0)"
        if not proj.attrs["net_capex_mode"]
        else "D&A omitted, so CapEx is treated as NET capex"
    )

    lines = []
    lines.append(f"**WACC used:** {wacc_label}")
    lines.append("")
    lines.append(f"**Horizon:** {years} years &nbsp;|&nbsp; "
                 f"**Terminal growth:** {terminal_g:.2%} &nbsp;|&nbsp; "
                 f"**Tax:** {tax_rate:.2%} &nbsp;|&nbsp; {capex_note}")
    lines.append("")
    lines.append("### Year-by-year projection ($B)")
    lines.append("```")
    lines.append(_table(proj))
    lines.append("```")

    lines.append("### Valuation bridge ($B)")
    lines.append("```")
    lines.append(_kpi(f"PV of explicit FCF (yrs 1-{years})", money_b(disc["pv_fcf"])))
    lines.append(_kpi("Terminal value (undiscounted)", money_b(tv)))
    lines.append(_kpi("PV of terminal value", money_b(disc["pv_tv"])))
    lines.append(_kpi("TV as % of EV", f"{disc['tv_pct_of_ev']:.2%}"))
    lines.append(_kpi("ENTERPRISE VALUE", money_b(disc["enterprise_value"])))
    lines.append(_kpi("Less net debt", money_b(bridge["net_debt"]))
                 + ("   (net cash: added back)" if bridge["net_cash"] else ""))
    lines.append(_kpi("EQUITY VALUE", money_b(bridge["equity_value"])))
    lines.append(_kpi("Shares outstanding", f"{shares:,.2f} B"))
    lines.append(_kpi("INTRINSIC VALUE / SHARE", money_ps(iv) + " per share"))
    lines.append(_kpi(f"Market price ({VALUATION_DATE})",
                      money_ps(current_price) + " per share"))
    lines.append(_kpi("UPSIDE / (DOWNSIDE)", f"{upside:.2%}"))
    lines.append("```")

    lines.append("### Break-even, case section 6.1 (steady-state perpetuity)")
    lines.append("```")
    lines.append(_kpi("EV = price x shares + net debt",
                      money_b(steady["target_enterprise_value"])))
    lines.append(_kpi("Required FCF = EV x (WACC - g)",
                      money_b(steady["required_fcf"])))
    lines.append(_kpi("FCF margin assumed", f"{steady['fcf_margin']:.2%}"))
    lines.append(_kpi("REQUIRED REVENUE", money_b(steady["required_rev"])))
    lines.append("")
    lines.append("Required revenue in context ($B):")
    for label, value in [
        ("FY26 actual (Ex 1)", REV_FY26),
        ("TTM", REV_TTM),
        ("Street FY27", REV_STREET_FY27),
    ]:
        lines.append(f"  {label:<24}{money_b(value):>12}"
                     f"{steady['required_rev'] / value:>10,.2f}x")
    lines.append("```")
    lines.append(
        f"To justify {money_ps(current_price)}, the section 6.1 run-rate revenue is "
        f"**{money_b(steady['required_rev'])}** at {steady['fcf_margin']:.0%} FCF "
        f"margin / {wacc:.4%} WACC / {terminal_g:.2%} g."
    )

    return (
        "\n".join(lines),
        scenario_panel(base_scenario, parse_scenario_values(scenario_values),
                       current_price),
        figure_projection(proj),
        figure_waterfall(disc, bridge, shares),
        figure_sensitivity(heatmap_mode, base_scenario, current_price),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HEADER = f"""
# NVIDIA DCF — teaching app

**Valuation date: {VALUATION_DATE}. Not investment advice.**

A teaching app for learning discounted cash flow mechanics. Every assumption
below is an input you choose, and the output is only as good as those inputs —
small changes to WACC, terminal growth, or the margin path move the answer a
great deal. Nothing here is a recommendation or a price target, and the author
is not a licensed investment adviser.

All figures in $ billions, shares in billions, rates as decimals (0.17 = 17%).
"""

FOOTNOTE = """
---
**Exhibit sources**

- **Exhibit 1** — FY2026 revenue, the base revenue the forecast grows from.
- **Exhibit 2B** — statutory / effective tax rate range of 16–18%; the 17%
  default is the midpoint.
- **Exhibit 4** — balance sheet debt and non-operating assets behind the net
  debt figure.
- **Exhibit 5** — FY26 gross capital expenditure and depreciation &
  amortisation, each as a percentage of revenue.
- **Exhibit 6** — shares outstanding, and non-operating assets net of debt (a
  net cash position, entered as a negative net debt).
- **Exhibit 7** — the market share price used as the valuation benchmark and as
  the input to the section 6.1 break-even.

Market price as of {date}. All math lives in `dcf.py`; this page only collects
inputs and formats what `dcf.py` returns.
""".format(date=VALUATION_DATE)


with gr.Blocks(title="NVIDIA DCF — teaching model") as demo:
    gr.Markdown(HEADER)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Company & market")
            base_revenue = gr.Number(value=DEFAULTS["base_revenue"],
                                     label="Base revenue, FY2026 ($B) — Ex 1")
            shares = gr.Number(value=DEFAULTS["shares"],
                               label="Shares outstanding (B) — Ex 6")
            net_debt = gr.Number(
                value=DEFAULTS["net_debt"],
                label="Net debt ($B) — Ex 4/6 (negative = net cash)")
            current_price = gr.Number(value=DEFAULTS["current_price"],
                                      label="Market price ($/share) — Ex 6/7")

            gr.Markdown("### Operating drivers")
            years = gr.Slider(1, 15, value=DEFAULTS["years"], step=1,
                              label="Forecast years (N)")
            tax_rate = gr.Number(value=DEFAULTS["tax_rate"],
                                 label="Tax rate — Ex 2B midpoint")
            capex_pct = gr.Number(value=DEFAULTS["capex_pct"],
                                  label="Gross CapEx % of revenue — Ex 5")
            da_pct = gr.Number(value=DEFAULTS["da_pct"],
                               label="D&A % of revenue — Ex 5")
            nwc_pct = gr.Number(
                value=DEFAULTS["nwc_pct"],
                label="NWC % of the CHANGE in revenue")

            gr.Markdown("### Discount rate")
            wacc_mode = gr.Radio(
                choices=["Direct WACC", "Build from CAPM"],
                value=DEFAULTS["wacc_mode"], label="WACC source")
            direct_wacc = gr.Number(value=DEFAULTS["direct_wacc"],
                                    label="Direct WACC")
            rf = gr.Number(value=DEFAULTS["rf"], label="Risk-free rate (CAPM)")
            erp = gr.Number(value=DEFAULTS["erp"], label="Equity risk premium (CAPM)")
            beta = gr.Number(value=DEFAULTS["beta"], label="Beta (CAPM)")
            after_tax_kd = gr.Number(value=DEFAULTS["after_tax_kd"],
                                     label="After-tax cost of debt")
            debt_weight = gr.Number(value=DEFAULTS["debt_weight"],
                                    label="Debt weight D/(D+E)")
            terminal_g = gr.Number(value=DEFAULTS["terminal_g"],
                                   label="Terminal growth g (must be < WACC)")

            gr.Markdown("### Revenue growth by year")
            growth_boxes = []
            for i in range(10):
                growth_boxes.append(
                    gr.Number(value=DEFAULTS["growth"][i], label=f"Growth Y{i + 1}")
                )
            long_term_growth = gr.Number(
                value=DEFAULTS["long_term_growth"],
                label="Long-term growth (years 11+ only)")

            gr.Markdown("### Operating margin by year")
            margin_boxes = []
            for i in range(10):
                margin_boxes.append(
                    gr.Number(value=DEFAULTS["margins"][i], label=f"Margin Y{i + 1}")
                )

            run = gr.Button("Run valuation", variant="primary")

        with gr.Column(scale=3):
            report = gr.Markdown()

    # scenario controls: five per scenario, the main sliders stay the live
    # "custom" case and supply everything a scenario does not override
    scenario_boxes = []
    with gr.Row():
        for _name, _defaults in SCENARIO_DEFAULTS:
            with gr.Column():
                gr.Markdown(f"#### {_name}")
                scenario_boxes += [
                    gr.Number(value=_defaults["prob"], label=f"{_name} probability"),
                    gr.Number(value=_defaults["y1_g"], label=f"{_name} year-1 growth"),
                    gr.Number(value=_defaults["terminal_g"], label=f"{_name} terminal g"),
                    gr.Number(value=_defaults["y10_m"], label=f"{_name} year-10 margin"),
                    gr.Number(value=_defaults["wacc"], label=f"{_name} WACC"),
                ]

    with gr.Row():
        scenario_report = gr.Markdown()

    # charts run the full width of the page, beneath the controls and text
    with gr.Row():
        projection_plot = gr.Plot(label="Revenue and FCF ($B)")
        waterfall_plot = gr.Plot(label="Valuation bridge ($B)")

    with gr.Row():
        heatmap_mode = gr.Dropdown(
            choices=[MODE_A, MODE_B], value=MODE_A,
            label="Sensitivity grid", scale=1)
    with gr.Row():
        sensitivity_plot = gr.Plot(label="IV/share sensitivity")

    gr.Markdown(FOOTNOTE)

    INPUTS = [
        base_revenue, shares, net_debt, current_price, years,
        tax_rate, capex_pct, da_pct, nwc_pct,
        wacc_mode, direct_wacc, rf, erp, beta, after_tax_kd, debt_weight,
        terminal_g,
        *growth_boxes, long_term_growth,
        *margin_boxes,
        heatmap_mode,
        *scenario_boxes,
    ]
    # text and all three charts come out of the one run_valuation call, so they
    # can never disagree with each other
    OUTPUTS = [report, scenario_report, projection_plot, waterfall_plot,
               sensitivity_plot]

    run.click(run_valuation, inputs=INPUTS, outputs=OUTPUTS)
    heatmap_mode.change(run_valuation, inputs=INPUTS, outputs=OUTPUTS)
    demo.load(run_valuation, inputs=INPUTS, outputs=OUTPUTS)


if __name__ == "__main__":
    demo.launch()
