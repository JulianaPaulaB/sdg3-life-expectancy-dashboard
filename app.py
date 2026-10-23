"""
SDG 3 Life Expectancy Dashboard (Streamlit)
Author: Juliana Paula T. Binas | BSIS 3A | Analytics Techniques and Tools

Files needed in the repo root: app.py, merged_sdg_data.csv, requirements.txt, .streamlit/config.toml
Main file on Streamlit Cloud: app.py
All figures (correlations, regression, R2, RMSE) are computed live from merged_sdg_data.csv.
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SDG 3 — Life Expectancy Dashboard",
                   page_icon="🫀", layout="wide")

# ── Colour palette ────────────────────────────────────────────────────────────
PURPLE, CYAN, ORANGE = "#9c27b0", "#00bcd4", "#ff9800"
INDIGO, TEAL, RED = "#3949ab", "#00897b", "#c62828"
GREEN, PINK = "#2e7d32", "#c2185b"
DARK, TEXT, GRID = "#1a1a2e", "#1a1a2e", "#dfe6ee"
UK_C, US_C = PURPLE, CYAN

# ── Data (live from CSV) ──────────────────────────────────────────────────────
YEARS = list(range(2000, 2025))
COLS = {"le": "Life Expectancy", "hc": "Healthcare Spending", "gdp": "GDP per capita",
        "co2": "CO2 Emissions", "water": "Water Access"}


@st.cache_data
def load_data():
    df = pd.read_csv(Path(__file__).parent / "merged_sdg_data.csv").dropna().drop_duplicates()
    df = df.rename(columns={v: k for k, v in COLS.items()})
    df["country"] = df["Country Name"].map({"United Kingdom": "UK", "United States": "USA"})
    return df


@st.cache_data
def fit_models(df):
    """Standardized OLS and Huber RLM (z-scored predictors and response)."""
    P = ["hc", "water", "gdp", "co2"]
    Z = (df[P] - df[P].mean()) / df[P].std(ddof=1)
    y = (df["le"] - df["le"].mean()) / df["le"].std(ddof=1)
    X = sm.add_constant(Z)
    rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    ols = sm.OLS(y, X).fit()
    fit = rlm.fittedvalues * df["le"].std(ddof=1) + df["le"].mean()
    res = df["le"] - fit
    r2 = 1 - (res ** 2).sum() / ((df["le"] - df["le"].mean()) ** 2).sum()
    return dict(beta=rlm.params[P], p=rlm.pvalues[P], r2=r2,
                rmse=float(np.sqrt((res ** 2).mean())), fit=fit, ols_r2=ols.rsquared,
                corr=df[P + ["le"]].corr()["le"][P], n=len(df))


df = load_data()
M = fit_models(df)
series = {c: {k: df[df.country == c].set_index("Year")[k].reindex(YEARS) for k in COLS}
          for c in ("UK", "USA")}

NAMES = {"hc": "Healthcare spending", "gdp": "GDP per capita",
         "co2": "CO₂ emissions", "water": "Water access"}
DRV_COL = {"hc": INDIGO, "gdp": ORANGE, "co2": RED, "water": TEAL}


def val(key, year, cnt):
    v = []
    if cnt != "USA": v.append(series["UK"][key][year])
    if cnt != "UK": v.append(series["USA"][key][year])
    v = [x for x in v if pd.notna(x)]
    return sum(v) / len(v) if v else None


def apply_base(fig, height=200, yrange=None, ysuffix=None, legend=False, xtitle=None, ytitle=None):
    """Consistent styling. All text is dark so labels stay readable on the light background."""
    fig.update_layout(
        height=height, margin=dict(l=0, r=8, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend, hovermode="closest",
        font=dict(size=11, color=TEXT, family="Segoe UI, system-ui, sans-serif"),
        legend=dict(font=dict(size=10, color=TEXT), orientation="h",
                    yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color=TEXT),
                     title=dict(text=xtitle, font=dict(size=10, color=TEXT)) if xtitle else None,
                     linecolor=GRID)
    ykw = dict(gridcolor=GRID, tickfont=dict(size=10, color=TEXT), linecolor=GRID,
               title=dict(text=ytitle, font=dict(size=10, color=TEXT)) if ytitle else None)
    if yrange: ykw["range"] = yrange
    if ysuffix: ykw["ticksuffix"] = ysuffix
    fig.update_yaxes(**ykw)


def banner_uri():
    """Header image: inline SVG (globe-style rings + pulse line), no external files needed."""
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='1400' height='170' viewBox='0 0 1400 170'>
<defs><linearGradient id='g' x1='0' x2='1'><stop offset='0' stop-color='#1a1a2e'/>
<stop offset='.55' stop-color='#3a1d5c'/><stop offset='1' stop-color='#00695c'/></linearGradient></defs>
<rect width='1400' height='170' fill='url(#g)'/>
<g fill='none' stroke='#ffffff' stroke-opacity='.10' stroke-width='2'>
<circle cx='1180' cy='85' r='120'/><circle cx='1180' cy='85' r='80'/><circle cx='1180' cy='85' r='40'/>
<ellipse cx='1180' cy='85' rx='120' ry='45'/><line x1='1060' y1='85' x2='1300' y2='85'/></g>
<polyline fill='none' stroke='#00bcd4' stroke-opacity='.65' stroke-width='3'
 points='0,120 180,120 215,120 235,70 262,150 285,95 305,120 560,120 595,120 615,78 640,146 662,100 680,120 900,120 935,120 955,82 980,142 1000,104 1020,120 1400,120'/>
</svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #eef3f8; }
  [data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 1rem !important; }
  .metric-card { background:#fff; border-radius:14px; border:1px solid #dfe6ee;
                 padding:14px 16px; margin-bottom:4px; }
  .insight-box { border-radius:0 10px 10px 0; padding:9px 12px; margin-top:8px;
                 font-size:12px; line-height:1.55; }
  .reco-card   { border-radius:12px; padding:13px 15px; margin-bottom:8px;
                 font-size:12px; line-height:1.55; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:url('{banner_uri()}') center/cover;padding:26px 30px;border-radius:14px;
            display:flex;align-items:center;gap:16px;margin-bottom:20px;flex-wrap:wrap;">
  <span style="background:#e8d5f5;border-radius:12px;padding:8px 14px;
               font-size:12px;font-weight:700;color:#6a0dad;">✦ SDG 3 · Good Health</span>
  <div style="margin-left:6px">
    <div style="color:#fff;font-size:20px;font-weight:800;">Drivers of Life Expectancy Across Countries</div>
    <div style="color:rgba(255,255,255,.8);font-size:11px;">
      Juliana Paula T. Binas · BSIS 3A · Analytics Techniques and Tools</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;">
    <span style="background:#e8d5f5;color:#6a0dad;border-radius:20px;padding:5px 12px;font-size:11px;font-weight:700;">2000–2024</span>
    <span style="background:#fce4ec;color:#c2185b;border-radius:20px;padding:5px 12px;font-size:11px;font-weight:700;">World Bank · Our World in Data</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{DARK};'>🎛️ Filters</h3>", unsafe_allow_html=True)
    year = st.slider("Year", 2000, 2024, 2023)
    country = st.selectbox("Country", ["Both countries", "United Kingdom", "United States"])
    cnt = "both" if country == "Both countries" else ("UK" if "Kingdom" in country else "USA")
    st.markdown("---")
    st.markdown(f"""<div style='font-size:12px;color:#444;line-height:1.8;'>
      <b>Data source:</b> World Bank, Our World in Data<br>
      <b>Model:</b> Robust RLM (Huber), standardized<br>
      <b>R²:</b> {M['r2']:.2f} &nbsp;|&nbsp; <b>RMSE:</b> {M['rmse']:.2f} yrs<br>
      <b>Observations:</b> {M['n']}<br>
      <span style='color:#888'>US data available 2005–2023</span></div>""", unsafe_allow_html=True)

if cnt == "USA" and pd.isna(series["USA"]["le"][year]):
    st.warning(f"No United States data for {year} (available 2005–2023). Pick another year.")

# ── KPI row ───────────────────────────────────────────────────────────────────
def kpi_card(key, label, icon, fmt, color, good_up=True):
    v = val(key, year, cnt)
    prev = val(key, year - 1, cnt) if year > 2000 else None
    if v is None:
        st.markdown(f"""<div class="metric-card" style="border-top:3px solid {color};">
          <div style="font-size:20px">{icon}</div>
          <div style="font-size:10px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:.5px;margin:4px 0 2px">{label}</div>
          <div style="font-size:16px;font-weight:700;color:#777">No data</div></div>""", unsafe_allow_html=True)
        return
    dv = v - prev if prev is not None else 0
    good = (dv > 0) == good_up
    arrow = "▲" if dv > 0 else ("▼" if dv < 0 else "—")
    dclr = "#777" if dv == 0 else (GREEN if good else "#d32f2f")   # green = favourable, red = unfavourable
    delta = f"{arrow} {fmt(abs(dv))} vs {year-1}" if prev is not None else "no prior year"
    st.markdown(f"""
    <div class="metric-card" style="border-top:3px solid {color};">
      <div style="font-size:20px">{icon}</div>
      <div style="font-size:10px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:.5px;margin:4px 0 2px">{label}</div>
      <div style="font-size:20px;font-weight:800;color:{DARK}">{fmt(v)}</div>
      <div style="font-size:10px;color:{dclr};font-weight:700;margin-top:4px">{delta}</div>
    </div>""", unsafe_allow_html=True)


k1, k2, k3, k4, k5 = st.columns(5)
with k1: kpi_card("le", "Life expectancy", "🫀", lambda v: f"{v:.1f} yrs", PURPLE, True)
with k2: kpi_card("hc", "Healthcare spending", "🏥", lambda v: f"${v:,.0f}", INDIGO, True)
with k3: kpi_card("gdp", "GDP per capita", "💰", lambda v: f"${v/1000:.0f}k", ORANGE, True)
with k4: kpi_card("co2", "CO₂ emissions", "🌫️", lambda v: f"{v:.2f} t", RED, False)
with k5: kpi_card("water", "Water access", "💧", lambda v: f"{v:.1f}%", TEAL, True)


def sec_head(dot, title):
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px;margin:22px 0 10px'>"
        f"<div style='width:9px;height:9px;border-radius:50%;background:{dot}'></div>"
        f"<span style='font-size:13px;font-weight:800;color:{DARK};text-transform:uppercase;"
        f"letter-spacing:.8px'>{title}</span></div>", unsafe_allow_html=True)


def insight(bg, border, color, text):
    st.markdown(f'<div class="insight-box" style="background:{bg};border-left:3px solid {border};'
                f'color:{color};">{text}</div>', unsafe_allow_html=True)


# ── Overview ──────────────────────────────────────────────────────────────────
sec_head(PURPLE, "Overview — trends over time")
col_trend, col_bar = st.columns([2, 1])

with col_trend:
    with st.container(border=True):
        st.markdown("**📈 Life expectancy over time**")
        traces = []
        if cnt != "USA":
            traces.append(go.Scatter(x=YEARS, y=series["UK"]["le"], name="United Kingdom",
                                     line=dict(color=UK_C, width=2.5), mode="lines+markers",
                                     marker=dict(size=4)))
        if cnt != "UK":
            traces.append(go.Scatter(x=YEARS, y=series["USA"]["le"], name="United States",
                                     line=dict(color=US_C, width=2.5, dash="dash"),
                                     mode="lines+markers", marker=dict(size=4), connectgaps=False))
        fig = go.Figure(traces)
        fig.add_vline(x=year, line=dict(color="#888", width=1, dash="dot"))
        apply_base(fig, 230, yrange=[75, 82.5], ysuffix=" yrs", legend=True)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        insight("#f9f0ff", "#ce93d8", "#4a148c",
                "✦ Both countries dipped in 2020–2021 (COVID-19). The UK recovered faster, and the USA "
                "has stayed below the UK in every year of the data despite far higher spending.")

with col_bar:
    with st.container(border=True):
        st.markdown(f"**📊 Country comparison — {year}**")
        lbls, vals, cols = [], [], []
        for c, name, colr in (("UK", "UK", UK_C), ("USA", "USA", US_C)):
            v = series[c]["le"][year]
            if cnt in ("both", c) and pd.notna(v):
                lbls.append(name); vals.append(v); cols.append(colr)
        fig = go.Figure(go.Bar(x=lbls, y=vals, marker=dict(color=cols),
                               text=[f"{v:.2f}" for v in vals], textposition="outside",
                               textfont=dict(color=TEXT)))
        apply_base(fig, 230, yrange=[70, 84], ysuffix=" yrs")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        uk_le, us_le = series["UK"]["le"][year], series["USA"]["le"][year]
        uk_hc, us_hc = series["UK"]["hc"][year], series["USA"]["hc"][year]
        if pd.notna(us_le) and pd.notna(uk_le):
            insight("#fce4ec", "#f48fb1", "#880e4f",
                    f"✦ In {year} the UK lives {uk_le-us_le:.1f} years longer while the USA spends "
                    f"{us_hc/uk_hc:.1f}× more per person — outcomes depend on more than total spending.")
        else:
            insight("#fce4ec", "#f48fb1", "#880e4f",
                    "✦ Pick a year between 2005 and 2023 to compare both countries.")

# ── Key Drivers ───────────────────────────────────────────────────────────────
sec_head(CYAN, "Key drivers — each variable vs life expectancy")
LIT = {"hc": "Jaba et al., 2014", "gdp": "Preston, 1975", "water": "UN, 2023", "co2": "Wen et al., 2021"}
NOTE = {
    "hc": "<b>Expected: positive.</b> Pooled, the USA spends most yet has lower life expectancy.",
    "gdp": "<b>Expected: positive.</b> Not significant in the model — it overlaps with healthcare spending.",
    "water": "<b>Expected: positive.</b> Almost no variation (98.9–100%), so the model cannot isolate it.",
    "co2": "<b>Expected: negative.</b> Strongest negative association with life expectancy.",
}
BG = {"hc": ("#e8eaf6", "#9fa8da"), "gdp": ("#fff3e0", "#ffcc80"),
      "water": ("#e0f2f1", "#80cbc4"), "co2": ("#ffebee", "#ef9a9a")}
FMT = {"hc": "$", "gdp": "$", "water": "%", "co2": " t"}

for col, key in zip(st.columns(4), ["hc", "gdp", "water", "co2"]):
    with col:
        with st.container(border=True):
            st.markdown(f"<b style='color:{DRV_COL[key]}'>{NAMES[key]}</b>", unsafe_allow_html=True)
            fig = go.Figure()
            for c, nm, colr in (("UK", "UK", UK_C), ("USA", "USA", US_C)):
                if cnt in ("both", c):
                    x, y = series[c][key], series[c]["le"]
                    fig.add_trace(go.Scatter(x=x, y=y, name=nm, mode="markers",
                                             marker=dict(color=colr, size=6, opacity=.55)))
                    if pd.notna(x[year]):
                        fig.add_trace(go.Scatter(x=[x[year]], y=[y[year]], name=f"{nm} {year}",
                                                 mode="markers", showlegend=False,
                                                 marker=dict(color=colr, size=13,
                                                             line=dict(color=DARK, width=2))))
            apply_base(fig, 190, yrange=[75.5, 82], legend=True, xtitle=NAMES[key], ytitle="Life exp. (yrs)")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            insight(*BG[key], DRV_COL[key], f"{NOTE[key]} ({LIT[key]})")

# ── Correlation ───────────────────────────────────────────────────────────────
sec_head(ORANGE, "Correlation analysis")
with st.container(border=True):
    st.markdown("**Pearson correlation with life expectancy (all 44 observations)**")
    order = M["corr"].sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        y=[NAMES[k] for k in order.index], x=order.values, orientation="h",
        marker=dict(color=[GREEN if v > 0 else RED for v in order.values]),
        text=[f"{v:+.2f}" for v in order.values], textposition="outside",
        textfont=dict(color=TEXT)))
    apply_base(fig, 200)
    fig.update_xaxes(range=[-1, 1], showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor="#888")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=TEXT))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    insight("#fff8e1", "#ffd54f", "#5d4037",
            "✦ Green = positive, red = negative. Healthcare spending and GDP correlate <b>negatively</b> "
            "with life expectancy only because the USA (high spending, lower life expectancy) is pooled "
            "with the UK. The regression below separates these overlapping effects.")

# ── Regression ────────────────────────────────────────────────────────────────
sec_head(PINK, "Regression model — performance &amp; standardized effects")
r1, r2c = st.columns(2)

with r1:
    with st.container(border=True):
        st.markdown("**📐 Model performance (Robust RLM)**")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² Score", f"{M['r2']:.2f}"); m2.metric("RMSE", f"{M['rmse']:.2f} yrs"); m3.metric("Obs.", f"{M['n']}")
        fig = go.Figure()
        lo, hi = df["le"].min() - .3, df["le"].max() + .3
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                                 line=dict(color="#888", dash="dot")))
        for c, nm, colr in (("UK", "UK", UK_C), ("USA", "USA", US_C)):
            m = df.country == c
            fig.add_trace(go.Scatter(x=df.loc[m, "le"], y=M["fit"][m], mode="markers", name=nm,
                                     text=df.loc[m, "Year"], marker=dict(color=colr, size=7, opacity=.8),
                                     hovertemplate="Year %{text}<br>Actual %{x:.2f}<br>Predicted %{y:.2f}"))
        apply_base(fig, 230, legend=True, xtitle="Actual life expectancy (yrs)", ytitle="Predicted (yrs)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        insight("#f3e5f5", "#ce93d8", "#4a148c",
                "✦ Points close to the dotted line are well predicted. The model explains about "
                f"{M['r2']*100:.0f}% of the variation in life expectancy.")

with r2c:
    with st.container(border=True):
        st.markdown("**📋 Standardized coefficients** *(comparable across variables)*")
        b, p = M["beta"].sort_values(), M["p"]
        fig = go.Figure(go.Bar(
            y=[NAMES[k] for k in b.index], x=b.values, orientation="h",
            marker=dict(color=[DRV_COL[k] if p[k] < .05 else "#b0b7c3" for k in b.index]),
            text=[f"{b[k]:+.2f}  (p {'<0.001' if p[k] < .001 else f'= {p[k]:.3f}'})" for k in b.index],
            textposition="outside", textfont=dict(color=TEXT)))
        apply_base(fig, 230)
        fig.update_xaxes(range=[-4.6, 3.6], showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor="#888",
                         title=dict(text="SD change in life expectancy per 1 SD change in predictor",
                                    font=dict(size=10, color=TEXT)))
        fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=TEXT))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        insight("#f3e5f5", "#ce93d8", "#4a148c",
                "✦ Coloured bars are significant (p &lt; 0.05); grey is not. CO₂ emissions is the largest "
                "effect. Betas beyond ±1 reflect overlap (multicollinearity) between predictors.")

# ── Recommendations ───────────────────────────────────────────────────────────
sec_head(GREEN, "Insights &amp; recommendations")
hc_beta = M["beta"]["hc"]
recos = [
    ("🌿 Reduce carbon emissions", RED, "#ffebee", "#b71c1c",
     "CO₂ per person is the strongest driver in the model and is significant (p &lt; 0.001). Cleaner energy "
     "and stricter emission standards are the most strongly supported actions in this data."),
    ("🫀 Invest in healthcare — and spend efficiently", INDIGO, "#e8eaf6", "#283593",
     f"Healthcare spending has a positive, significant effect under robust regression (beta {hc_beta:+.2f}, "
     "p = 0.036), but the USA spends about 2× more than the UK with shorter lives. Value for money matters as much as the total."),
    ("🏦 Look beyond GDP alone", "#e65100", "#fff3e0", "#bf360c",
     "GDP per capita is not significant once healthcare spending and CO₂ are included. Growth helps mainly "
     "when it translates into health services and cleaner environments."),
    ("💧 Water access: collect more country data", "#004d40", "#e0f2f1", "#00695c",
     "Water access shows a negative sign only because the UK and USA are both near 100%. This is a data limit, "
     "not evidence against clean water. Adding lower-access countries would test this properly."),
]
for i in range(0, 4, 2):
    ca, cb = st.columns(2)
    for col, rc in zip([ca, cb], recos[i:i + 2]):
        with col:
            st.markdown(f'<div class="reco-card" style="background:{rc[2]};border:1px solid #dfe6ee;">'
                        f'<b style="color:{rc[1]};font-size:13px;">{rc[0]}</b><br>'
                        f'<span style="color:{rc[3]}">{rc[4]}</span></div>', unsafe_allow_html=True)

# ── Conclusion ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{DARK};border-radius:14px;padding:20px 24px;margin-top:14px;color:#fff;">
  <h3 style="font-size:15px;font-weight:800;color:#e8d5f5;margin-bottom:8px;">✦ Final conclusion</h3>
  <p style="font-size:12.5px;line-height:1.75;color:rgba(255,255,255,.9);">
    The robust regression shows that <b>CO₂ emissions</b> (negative) and <b>healthcare spending</b>
    (positive) are significant drivers of life expectancy, in line with the literature.
    <b>GDP per capita</b> adds no independent effect, and the negative <b>water access</b> result reflects
    near-zero variation in the UK and USA data. Together the variables explain about
    <b>{M['r2']*100:.0f}%</b> of the variation in life expectancy across the UK and USA (2000–2024),
    supporting the goals of <b>SDG 3: Good Health and Well-Being</b>.
    Limits: two countries, 44 observations, and results show association, not causation.
  </p>
  <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
    <span style="background:{RED};color:#fff;border-radius:20px;padding:5px 13px;font-size:11px;font-weight:700;">CO₂ emissions ↓</span>
    <span style="background:{INDIGO};color:#fff;border-radius:20px;padding:5px 13px;font-size:11px;font-weight:700;">Healthcare spending ↑</span>
    <span style="background:#757575;color:#fff;border-radius:20px;padding:5px 13px;font-size:11px;font-weight:700;">GDP per capita (n.s.)</span>
    <span style="background:#e8d5f5;color:#4a148c;border-radius:20px;padding:5px 13px;font-size:11px;font-weight:700;">R² = {M['r2']:.2f}</span>
  </div>
</div>
""", unsafe_allow_html=True)
