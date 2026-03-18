"""
PontIA · Marketing Intelligence Dashboard
──────────────────────────────────────────
Cuadro de mandos semanal de campañas de marketing.
Conectado a Google Sheets (o Excel local como fallback).
"""

import hashlib
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from utils.data_loader import apply_filters, get_filter_options, load_data

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PontIA · Marketing Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# BRAND PALETTE
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":      "#111E2D",
    "bg2":     "#0E1A26",
    "card":    "#162535",
    "green":   "#173A32",
    "yellow":  "#F6FAB2",
    "amber":   "#BB812F",
    "orange":  "#EE7015",
    "blue":    "#5683D2",
    "sage":    "#AABCA3",
    "red":     "#6C0000",
    "purple":  "#744A6E",
    "ok":      "#4CAF50",
    "warn":    "#FFC107",
    "danger":  "#EF5350",
    "muted":   "#8BA0B0",
    "border":  "#1E3347",
}

CHANNEL_COLORS = {
    "Meta Ads (FB/IG)": C["orange"],
    "Google Ads":       C["blue"],
    "Orgánico / SEO":   C["sage"],
    "YouTube Ads":      C["amber"],
    "LinkedIn Ads":     C["purple"],
}

CHART_PALETTE = [
    C["blue"], C["orange"], C["yellow"], C["sage"],
    C["amber"], C["purple"], "#EF5350", C["ok"],
    "#00BCD4", "#FF80AB", "#FFB74D", "#81C784",
]

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,37,53,0.4)",
    font=dict(family="Manrope, sans-serif", color="#FFFFFF", size=12),
    title_font=dict(family="Manrope, sans-serif", color=C["yellow"], size=14),
    legend=dict(
        bgcolor="rgba(14,26,38,0.8)",
        bordercolor=C["border"],
        borderwidth=1,
        font=dict(color="#FFFFFF", size=11),
    ),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"], tickfont=dict(color=C["muted"]), zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"], tickfont=dict(color=C["muted"]), zerolinecolor=C["border"]),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor=C["card"], bordercolor=C["border"], font_family="Manrope", font_color="#FFFFFF"),
)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ── Global ────────────────────────────────── */
html,body,[class*="css"]{font-family:'Manrope',-apple-system,sans-serif!important}
.stApp{background:#111E2D}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.2rem;padding-bottom:1rem;max-width:1500px}
a{color:#F6FAB2}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0E1A26 0%,#0A1520 100%);border-right:1px solid #1E3347}
[data-testid="stSidebar"] .block-container{padding-top:.8rem}

/* ── Tabs ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{background:#0E1A26;border-radius:10px;padding:4px;gap:4px;border:1px solid #1E3347;margin-bottom:.8rem}
.stTabs [data-baseweb="tab"]{background:transparent;color:#8BA0B0;border-radius:7px;padding:.45rem 1rem;font-family:'Manrope',sans-serif;font-weight:500;font-size:.875rem;transition:all .2s;border:none}
.stTabs [aria-selected="true"]{background:#173A32!important;color:#F6FAB2!important;font-weight:700}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}

/* ── KPI Cards ────────────────────────────── */
.kpi-card{background:linear-gradient(135deg,#162535,#1A2D3D);border:1px solid #1E3347;border-radius:14px;padding:1.1rem .9rem;text-align:center;transition:all .25s;position:relative;overflow:hidden}
.kpi-card::before{content:'';position:absolute;top:0;left:0;width:100%;height:3px;background:linear-gradient(90deg,#F6FAB2,#173A32)}
.kpi-card:hover{border-color:#F6FAB2;transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,.4)}
.kpi-icon{font-size:1.3rem;margin-bottom:.25rem}
.kpi-label{font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#8BA0B0;margin-bottom:.35rem}
.kpi-value{font-family:'IBM Plex Mono',monospace;font-size:1.65rem;font-weight:600;color:#F6FAB2;line-height:1;margin-bottom:.25rem}
.kpi-sub{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#8BA0B0}
.badge{display:inline-block;font-size:.62rem;font-weight:700;padding:.12rem .45rem;border-radius:20px;margin-top:.25rem}
.bg{background:rgba(76,175,80,.18);color:#4CAF50;border:1px solid rgba(76,175,80,.3)}
.by{background:rgba(255,193,7,.18);color:#FFC107;border:1px solid rgba(255,193,7,.3)}
.br{background:rgba(239,83,80,.18);color:#EF5350;border:1px solid rgba(239,83,80,.3)}

/* ── Section title ────────────────────────── */
.sec{font-size:.8rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#F6FAB2;border-left:3px solid #F6FAB2;padding-left:.7rem;margin:1.2rem 0 .6rem}

/* ── Alert boxes ──────────────────────────── */
.al{border-radius:9px;padding:.75rem .9rem;font-size:.85rem;margin:.3rem 0;display:flex;align-items:flex-start;gap:.5rem;line-height:1.4}
.al-w{background:rgba(238,112,21,.12);border:1px solid rgba(238,112,21,.4);color:#FFA07A}
.al-d{background:rgba(239,83,80,.12);border:1px solid rgba(239,83,80,.4);color:#EF9A9A}
.al-s{background:rgba(76,175,80,.12);border:1px solid rgba(76,175,80,.4);color:#A5D6A7}
.al-i{background:rgba(86,131,210,.12);border:1px solid rgba(86,131,210,.4);color:#90CAF9}

/* ── Buttons ──────────────────────────────── */
.stButton>button{background:linear-gradient(135deg,#173A32,#1E4A3E)!important;color:#F6FAB2!important;border:1px solid #1E3347!important;border-radius:8px!important;font-family:'Manrope',sans-serif!important;font-weight:700!important;font-size:.875rem!important;padding:.55rem 1.4rem!important;transition:all .2s!important}
.stButton>button:hover{border-color:#F6FAB2!important;transform:translateY(-1px)}

/* ── Inputs ───────────────────────────────── */
.stSelectbox [data-baseweb="select"],.stMultiSelect [data-baseweb="select"]{background:#162535;border-color:#1E3347;border-radius:8px}
.stTextInput input,.stPasswordInput input{background:#162535!important;border-color:#1E3347!important;border-radius:8px!important;color:#FFF!important}

/* ── Login card ───────────────────────────── */
.login-card{background:linear-gradient(135deg,#162535,#1A2D3D);border:1px solid #1E3347;border-radius:20px;padding:2.5rem 2rem;box-shadow:0 20px 60px rgba(0,0,0,.5)}
.login-title{font-family:'Manrope',sans-serif;font-size:1.6rem;font-weight:800;color:#F6FAB2;text-align:center;margin-bottom:.2rem}
.login-sub{font-size:.85rem;color:#8BA0B0;text-align:center;margin-bottom:1.8rem}

/* ── Sidebar brand ────────────────────────── */
.sb-brand{text-align:center;padding:.5rem 0 1.2rem;border-bottom:1px solid #1E3347;margin-bottom:.8rem}
.sb-title{font-family:'Manrope',sans-serif;font-size:1.35rem;font-weight:800;color:#F6FAB2}
.sb-sub{font-size:.7rem;color:#8BA0B0;letter-spacing:.1em;text-transform:uppercase;margin-top:.1rem}
.sb-user{background:rgba(23,58,50,.3);border:1px solid #1E3347;border-radius:8px;padding:.5rem .7rem;font-size:.78rem;color:#AABCA3;margin-bottom:.8rem}

/* ── Divider ──────────────────────────────── */
.div{height:1px;background:linear-gradient(90deg,#1E3347,transparent);margin:.8rem 0}

/* ── DataFrames ───────────────────────────── */
.stDataFrame{border-radius:10px;overflow:hidden}
</style>
""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════
def _load_auth_config() -> dict:
    """Lee credenciales de st.secrets o config.yaml (fallback)."""
    try:
        pw = st.secrets["auth"]["password"]
        emails = list(st.secrets["auth"]["authorized_emails"])
        return {"password": pw, "authorized_emails": emails}
    except Exception:
        pass
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("auth", {})
    return {}


def check_login(email: str, password: str) -> bool:
    cfg = _load_auth_config()
    authorized = [e.lower().strip() for e in cfg.get("authorized_emails", [])]
    return email.lower().strip() in authorized and password == cfg.get("password", "")


def show_login_page():
    inject_css()
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align:center;margin-bottom:1.5rem">
                <span style="font-size:2.8rem">🧠</span><br>
                <span class="login-title">PontIA</span><br>
                <span class="login-sub">Marketing Intelligence · Acceso privado</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        email = st.text_input("Correo electrónico", placeholder="tu@pontia.es", key="login_email")
        password = st.text_input("Contraseña", type="password", placeholder="••••••••", key="login_pw")
        if st.button("Entrar →", use_container_width=True):
            if check_login(email, password):
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = email
                st.rerun()
            else:
                st.error("Correo o contraseña incorrectos. Contacta con el administrador.")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fmt_eur(v, dec=0):
    if pd.isna(v):
        return "—"
    return f"{v:,.{dec}f} €".replace(",", "·")


def fmt_pct(v, dec=1):
    if pd.isna(v):
        return "—"
    return f"{v * 100:.{dec}f}%"


def fmt_num(v, dec=0):
    if pd.isna(v):
        return "—"
    return f"{v:,.{dec}f}".replace(",", "·")


def _color_badge(value, good_thresh, bad_thresh, invert=False):
    """Devuelve clase CSS del badge según umbral (invert=True para métricas donde menor es mejor)."""
    if pd.isna(value):
        return ""
    if invert:
        if value <= good_thresh:
            return "bg"
        elif value <= bad_thresh:
            return "by"
        return "br"
    else:
        if value >= good_thresh:
            return "bg"
        elif value >= bad_thresh:
            return "by"
        return "br"


def kpi_card(icon, label, value, sub="", badge_class="", badge_text=""):
    badge_html = f'<div class="badge {badge_class}">{badge_text}</div>' if badge_class else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def alert(text, kind="i"):
    icons = {"w": "⚠️", "d": "🔴", "s": "✅", "i": "ℹ️"}
    st.markdown(f'<div class="al al-{kind}">{icons[kind]} {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def _apply_base(fig, title=""):
    fig.update_layout(**PLOT_BASE, title_text=title)
    return fig


def chart_evolucion_semanal(df: pd.DataFrame):
    """Evolución semanal de KPIs globales agrupados por semana."""
    g = (
        df.groupby("Semana_label")
        .agg(
            Inversión=("Inversión (€)", "sum"),
            Leads=("Leads Válidos", "sum"),
            Entrevistas=("Entrevistas", "sum"),
            Matriculados=("Matriculados", "sum"),
        )
        .reset_index()
    )
    # Ordenar semanas numéricamente
    g["_ord"] = g["Semana_label"].str.replace("S", "").astype(int)
    g = g.sort_values("_ord")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(name="Inversión (€)", x=g["Semana_label"], y=g["Inversión"],
               marker_color=C["green"], opacity=0.85,
               hovertemplate="<b>%{x}</b><br>Inversión: %{y:,.0f} €<extra></extra>"),
        secondary_y=False,
    )
    for col, color, dash in [
        ("Leads", C["yellow"], "solid"),
        ("Entrevistas", C["orange"], "dot"),
        ("Matriculados", C["ok"], "dashdot"),
    ]:
        fig.add_trace(
            go.Scatter(name=col, x=g["Semana_label"], y=g[col],
                       mode="lines+markers", line=dict(color=color, width=2.5, dash=dash),
                       marker=dict(size=7, symbol="circle"),
                       hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y}}<extra></extra>"),
            secondary_y=True,
        )
    fig.update_layout(**PLOT_BASE, title_text="Evolución Semanal — Inversión & Resultados")
    fig.update_layout(legend=dict(**PLOT_BASE["legend"], orientation="h", y=-0.15))
    fig.update_yaxes(title_text="Inversión (€)", secondary_y=False,
                     title_font=dict(color=C["sage"]), tickfont=dict(color=C["muted"]),
                     gridcolor=C["border"])
    fig.update_yaxes(title_text="Leads / Entrevistas / Matrículas", secondary_y=True,
                     title_font=dict(color=C["yellow"]), tickfont=dict(color=C["muted"]),
                     gridcolor=C["border"])
    return fig


def chart_roas_campanas(df: pd.DataFrame):
    """ROAS por campaña (barras horizontales, ordenadas)."""
    g = (
        df[df["Inversión (€)"].fillna(0) > 0]
        .groupby("ID_Campaña")
        .agg(Ingresos=("Ingresos (€)", "sum"), Inversión=("Inversión (€)", "sum"))
        .reset_index()
    )
    g["ROAS"] = g["Ingresos"] / g["Inversión"]
    g = g.sort_values("ROAS", ascending=True).tail(15)
    colors = [C["ok"] if r >= 4 else C["warn"] if r >= 1 else C["danger"] for r in g["ROAS"]]
    fig = go.Figure(go.Bar(
        x=g["ROAS"], y=g["ID_Campaña"], orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>ROAS: %{x:.2f}x<extra></extra>",
        text=g["ROAS"].apply(lambda x: f"{x:.2f}x"),
        textposition="outside", textfont=dict(color=C["yellow"], size=11),
    ))
    fig.add_vline(x=1, line_dash="dash", line_color=C["warn"], annotation_text="Break-even",
                  annotation_font=dict(color=C["warn"], size=10))
    fig.update_layout(**PLOT_BASE, title_text="ROAS por Campaña",
                      xaxis=dict(**PLOT_BASE["xaxis"], title="ROAS"),
                      height=max(300, len(g) * 36 + 80))
    return fig


def chart_cpl_campanas(df: pd.DataFrame):
    """CPL y Leads Válidos por campaña (barras + scatter)."""
    g = (
        df.groupby("ID_Campaña")
        .agg(
            CPL=("CPL (€)", "mean"),
            Leads=("Leads Válidos", "sum"),
            Inversión=("Inversión (€)", "sum"),
        )
        .reset_index()
        .dropna(subset=["CPL"])
    )
    g = g[g["CPL"] > 0].sort_values("CPL")
    colors = [C["ok"] if v <= 15 else C["warn"] if v <= 30 else C["danger"] for v in g["CPL"]]
    fig = go.Figure(go.Bar(
        x=g["CPL"], y=g["ID_Campaña"], orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>CPL: %{x:.2f} €<br>Leads: %{customdata}<extra></extra>",
        customdata=g["Leads"],
        text=g["CPL"].apply(lambda x: f"{x:.1f} €"),
        textposition="outside", textfont=dict(color=C["muted"], size=10),
    ))
    fig.add_vline(x=15, line_dash="dash", line_color=C["ok"], annotation_text="Óptimo ≤15€",
                  annotation_font=dict(color=C["ok"], size=10))
    fig.update_layout(**PLOT_BASE, title_text="CPL Medio por Campaña (€)",
                      xaxis=dict(**PLOT_BASE["xaxis"], title="CPL (€)"),
                      height=max(300, len(g) * 36 + 80))
    return fig


def chart_distribucion_canal(df: pd.DataFrame):
    """Inversión y leads por canal (donut)."""
    g = df.groupby("Canal").agg(
        Inversión=("Inversión (€)", "sum"),
        Leads=("Leads Válidos", "sum"),
    ).reset_index()
    g = g[g["Inversión"] > 0]

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "domain"}, {"type": "domain"}]],
                        subplot_titles=["Inversión por canal", "Leads por canal"])
    for i, (col, title) in enumerate([("Inversión", "Inversión"), ("Leads", "Leads")], 1):
        clrs = [CHANNEL_COLORS.get(c, C["muted"]) for c in g["Canal"]]
        fig.add_trace(
            go.Pie(labels=g["Canal"], values=g[col], name=title,
                   hole=0.55, marker_colors=clrs,
                   textinfo="percent+label", textfont_size=10,
                   hovertemplate=f"<b>%{{label}}</b><br>{title}: %{{value:,.0f}}<br>%{{percent}}<extra></extra>"),
            row=1, col=i,
        )
    fig.update_layout(**PLOT_BASE, title_text="Distribución por Canal",
                      showlegend=False, height=320)
    return fig


def chart_mapa_eficiencia(df: pd.DataFrame):
    """Scatter: CPL vs Leads Válidos, tamaño = Inversión."""
    g = (
        df.groupby("ID_Campaña")
        .agg(
            CPL=("CPL (€)", "mean"),
            Leads=("Leads Válidos", "sum"),
            Inversión=("Inversión (€)", "sum"),
            Canal=("Canal", "first"),
            ROAS=("ROAS", "mean"),
        )
        .reset_index()
        .dropna(subset=["CPL"])
    )
    g = g[(g["CPL"] > 0) & (g["Leads"] > 0)]
    g["color"] = g["Canal"].map(lambda x: CHANNEL_COLORS.get(x, C["muted"]))
    g["size_px"] = (g["Inversión"].fillna(0) ** 0.5).clip(8, 50)

    fig = go.Figure()
    for canal in g["Canal"].unique():
        sub = g[g["Canal"] == canal]
        fig.add_trace(go.Scatter(
            x=sub["CPL"], y=sub["Leads"],
            mode="markers+text",
            name=canal,
            marker=dict(
                size=sub["size_px"], color=CHANNEL_COLORS.get(canal, C["muted"]),
                line=dict(width=1.5, color="rgba(255,255,255,0.3)"), opacity=0.85,
            ),
            text=sub["ID_Campaña"].str[:14],
            textposition="top center",
            textfont=dict(size=8, color=C["muted"]),
            hovertemplate=(
                "<b>%{text}</b><br>CPL: %{x:.2f} €<br>Leads: %{y}"
                "<br>Inversión: %{customdata:,.0f} €<extra></extra>"
            ),
            customdata=sub["Inversión"],
        ))
    fig.add_vline(x=15, line_dash="dash", line_color=C["ok"], opacity=0.5)
    fig.update_layout(**PLOT_BASE, title_text="Mapa de Eficiencia — CPL vs Leads (tamaño = Inversión)",
                      xaxis=dict(**PLOT_BASE["xaxis"], title="CPL (€) — menor es mejor"),
                      yaxis=dict(**PLOT_BASE["yaxis"], title="Leads Válidos"),
                      height=480)
    return fig


def chart_alta_intencion(df: pd.DataFrame):
    """% Alta Intención por semana y canal."""
    g = (
        df.groupby(["Semana_label", "Canal"])
        .agg(
            Consideracion=("Consideración", "sum"),
            Decision=("Decisión", "sum"),
            Leads=("Leads Válidos", "sum"),
        )
        .reset_index()
    )
    g = g[g["Leads"] > 0]
    g["% Alta Int."] = (g["Consideracion"] + g["Decision"]) / g["Leads"] * 100
    g["_ord"] = g["Semana_label"].str.replace("S", "").astype(int)
    g = g.sort_values("_ord")

    fig = px.line(
        g, x="Semana_label", y="% Alta Int.", color="Canal",
        color_discrete_map=CHANNEL_COLORS,
        markers=True, line_shape="spline",
        hover_data={"Leads": True},
    )
    fig.update_traces(line_width=2.5, marker_size=7)
    fig.update_layout(**PLOT_BASE, title_text="% Leads de Alta Intención por Semana y Canal",
                      yaxis=dict(**PLOT_BASE["yaxis"], title="% Alta Intención", ticksuffix="%"))
    fig.update_layout(legend=dict(**PLOT_BASE["legend"], orientation="h", y=-0.2))
    return fig


def chart_embudo(df: pd.DataFrame):
    """Embudo de conversión global."""
    totals = {
        "Contactos":      df["Contactos"].sum(),
        "Leads Válidos":  df["Leads Válidos"].sum(),
        "Entrevistas":    df["Entrevistas"].sum(),
        "Matriculados":   df["Matriculados"].sum(),
    }
    labels = list(totals.keys())
    values = list(totals.values())
    pcts = [f"{v/values[0]*100:.1f}%" if values[0] > 0 else "" for v in values]

    fig = go.Figure(go.Funnel(
        name="Embudo",
        y=labels, x=values,
        textinfo="value+percent initial",
        marker_color=[C["blue"], C["yellow"], C["orange"], C["ok"]],
        connector=dict(line=dict(color=C["border"], width=1)),
        hovertemplate="<b>%{y}</b><br>Total: %{x}<br>Del total: %{percentInitial}<extra></extra>",
    ))
    fig.update_layout(**PLOT_BASE, title_text="Embudo de Conversión Global", height=320)
    return fig


def chart_motivos_perdida(df: pd.DataFrame):
    """Desglose de motivos de pérdida (donut)."""
    loss_cols = {
        "No válido":               "No válido",
        "No es lo que buscaba":    "No es lo que buscaba",
        "No tiene dinero":         "Precio",
        "No interesa (Otros)":     "No interesa",
        "Matriculado otra escuela":"Competencia",
        "Próxima convocatoria":    "Próxima conv.",
        "Busca Certificación":     "Certif.",
        "Busca otra metodología":  "Otra metodología",
        "Ilocalizado":             "Ilocalizable",
        "No tiene tiempo":         "Sin tiempo",
        "Desistimiento":           "Desistimiento",
    }
    vals, lbls = [], []
    for col, label in loss_cols.items():
        if col in df.columns:
            v = df[col].sum()
            if v > 0:
                vals.append(v)
                lbls.append(label)

    if not vals:
        return None

    palette = [C["orange"], C["blue"], C["danger"], C["amber"], C["purple"],
               C["sage"], C["warn"], C["ok"], "#00BCD4", "#FF80AB", "#FFB74D"]
    fig = go.Figure(go.Pie(
        labels=lbls, values=vals, hole=0.52,
        marker_colors=palette[:len(lbls)],
        textinfo="percent+label", textfont_size=10,
        hovertemplate="<b>%{label}</b><br>Cantidad: %{value}<br>%{percent}<extra></extra>",
        pull=[0.04 if v == max(vals) else 0 for v in vals],
    ))
    fig.update_layout(**PLOT_BASE, title_text="Motivos de Pérdida de Leads", height=350, showlegend=False)
    return fig


def chart_heatmap_campanas(df: pd.DataFrame, metric="CPL (€)"):
    """Heatmap: Campañas × Semanas con color = métrica."""
    pivot = df.pivot_table(index="ID_Campaña", columns="Semana_label", values=metric, aggfunc="mean")
    if pivot.empty:
        return None
    # Ordenar columnas semana
    cols = sorted(pivot.columns, key=lambda x: int(x.replace("S", "")))
    pivot = pivot[cols]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0, C["ok"]], [0.5, C["warn"]], [1, C["danger"]]],
        hovertemplate="<b>%{y}</b><br>Semana: %{x}<br>Valor: %{z:.2f}<extra></extra>",
        colorbar=dict(title=metric, tickfont=dict(color=C["muted"]), title_font=dict(color=C["muted"])),
    ))
    fig.update_layout(**PLOT_BASE, title_text=f"Heatmap — {metric} por Campaña × Semana",
                      yaxis=dict(**PLOT_BASE["yaxis"], tickfont=dict(size=10, color=C["muted"])),
                      height=max(300, len(pivot) * 30 + 100))
    return fig


def chart_evolucion_campana(df: pd.DataFrame, metric="Leads Válidos"):
    """Evolución de una métrica por campaña a lo largo de las semanas."""
    pivot = df.pivot_table(index="Semana_label", columns="ID_Campaña", values=metric, aggfunc="sum")
    if pivot.empty:
        return None
    pivot = pivot.loc[sorted(pivot.index, key=lambda x: int(x.replace("S", "")))]

    fig = go.Figure()
    for i, camp in enumerate(pivot.columns):
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[camp], name=camp[:20],
            mode="lines+markers", line=dict(color=color, width=2),
            marker=dict(size=6), connectgaps=True,
            hovertemplate=f"<b>{camp}</b><br>Semana: %{{x}}<br>{metric}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(**PLOT_BASE, title_text=f"Evolución de {metric} por Campaña", height=420)
    fig.update_layout(legend=dict(**PLOT_BASE["legend"], orientation="h", y=-0.2))
    return fig


def chart_perdida_por_semana(df: pd.DataFrame):
    """Evolución semanal de leads perdidos vs entrevistados."""
    g = (
        df.groupby("Semana_label")
        .agg(
            Perdidos=("Perdidos", "sum"),
            Entrevistas=("Entrevistas", "sum"),
            Matriculados=("Matriculados", "sum"),
        )
        .reset_index()
    )
    g["_ord"] = g["Semana_label"].str.replace("S", "").astype(int)
    g = g.sort_values("_ord")

    fig = go.Figure()
    for col, color in [("Entrevistas", C["blue"]), ("Matriculados", C["ok"]), ("Perdidos", C["danger"])]:
        fig.add_trace(go.Bar(name=col, x=g["Semana_label"], y=g[col], marker_color=color, opacity=0.85))
    fig.update_layout(**PLOT_BASE, title_text="Entrevistas, Matrículas y Pérdidas por Semana",
                      barmode="group")
    fig.update_layout(legend=dict(**PLOT_BASE["legend"], orientation="h", y=-0.15))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
def tab_resumen(df: pd.DataFrame, df_all: pd.DataFrame):
    """Tab 1: Resumen general con KPIs y gráficas clave."""
    # ── KPIs ──────────────────────────────────────────────────────────────────
    section("KPIs Globales")
    inv = df["Inversión (€)"].sum()
    leads = int(df["Leads Válidos"].sum())
    cpl = df["CPL (€)"].mean()
    entrevistas = int(df["Entrevistas"].sum())
    coste_ent = inv / entrevistas if entrevistas > 0 else None
    matriculados = int(df["Matriculados"].sum())
    ingresos = df["Ingresos (€)"].sum()
    roas = ingresos / inv if inv > 0 else None
    conv_mat = matriculados / leads if leads > 0 else None

    cols = st.columns(8)
    with cols[0]:
        kpi_card("💰", "Inversión Total", fmt_eur(inv), "acumulado")
    with cols[1]:
        kpi_card("📥", "Leads Válidos", fmt_num(leads), "total",
                 _color_badge(leads, 100, 50), f"{'▲' if leads > 100 else '▼'} {leads}")
    with cols[2]:
        badge = _color_badge(cpl, 15, 30, invert=True) if cpl else ""
        kpi_card("💡", "CPL Medio", fmt_eur(cpl, 2), "coste por lead", badge,
                 "≤15€ óptimo" if badge == "bg" else ">30€ alto" if badge == "br" else "aceptable")
    with cols[3]:
        kpi_card("🤝", "Entrevistas", fmt_num(entrevistas), "total")
    with cols[4]:
        badge = _color_badge(coste_ent, 60, 100, invert=True) if coste_ent else ""
        kpi_card("💼", "Coste/Entrevista", fmt_eur(coste_ent, 1) if coste_ent else "—", "€ por entrevista",
                 badge, "")
    with cols[5]:
        kpi_card("🎓", "Matriculados", fmt_num(matriculados), "total",
                 _color_badge(matriculados, 5, 2), "")
    with cols[6]:
        kpi_card("💵", "Ingresos", fmt_eur(ingresos), "acumulado")
    with cols[7]:
        badge = _color_badge(roas, 3, 1) if roas else ""
        kpi_card("📈", "ROAS Global", f"{roas:.2f}x" if roas else "—", "retorno inversión",
                 badge, "≥3x bueno" if badge == "bg" else "<1x negativo" if badge == "br" else "")

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Alertas automáticas ────────────────────────────────────────────────
    campanas_malas = (
        df[df["Inversión (€)"].fillna(0) > 0]
        .groupby("ID_Campaña")
        .agg(Leads=("Leads Válidos", "sum"), Inv=("Inversión (€)", "sum"), Matr=("Matriculados", "sum"))
        .reset_index()
    )
    campanas_malas["CPL_calc"] = campanas_malas["Inv"] / campanas_malas["Leads"].replace(0, np.nan)
    sin_leads = campanas_malas[campanas_malas["Leads"] == 0]["ID_Campaña"].tolist()
    cpl_alto = campanas_malas[campanas_malas["CPL_calc"] > 40]["ID_Campaña"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        if sin_leads:
            alert(f"<b>Sin leads:</b> {', '.join(sin_leads[:3])}{'…' if len(sin_leads)>3 else ''} — considera pausar.", "d")
        if cpl_alto:
            alert(f"<b>CPL elevado (>40€):</b> {', '.join(cpl_alto[:3])}{'…' if len(cpl_alto)>3 else ''}", "w")
    with c2:
        if leads > 0 and conv_mat:
            alert(f"Tasa de conversión Lead→Matrícula: <b>{fmt_pct(conv_mat)}</b> — benchmark sector ~2-5%.", "i")

    # ── Gráficas ───────────────────────────────────────────────────────────
    section("Evolución Semanal")
    fig_evo = chart_evolucion_semanal(df_all)
    st.plotly_chart(fig_evo, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        section("ROAS por Campaña")
        st.plotly_chart(chart_roas_campanas(df), use_container_width=True, config={"displayModeBar": False})
    with c2:
        section("Distribución por Canal")
        st.plotly_chart(chart_distribucion_canal(df), use_container_width=True, config={"displayModeBar": False})


def tab_campanas(df: pd.DataFrame):
    """Tab 2: Análisis detallado por campaña."""
    section("Tabla de Rendimiento por Campaña")

    summary = (
        df.groupby(["ID_Campaña", "Canal", "Programa"])
        .agg(
            Inversión=("Inversión (€)", "sum"),
            Leads=("Leads Válidos", "sum"),
            CPL=("CPL (€)", "mean"),
            Entrevistas=("Entrevistas", "sum"),
            Coste_Ent=("Coste Entrevista (€)", "mean"),
            Matriculados=("Matriculados", "sum"),
            Ingresos=("Ingresos (€)", "sum"),
            Alta_Int=("% Alta Intención", "mean"),
        )
        .reset_index()
    )
    summary["ROAS"] = np.where(summary["Inversión"] > 0, summary["Ingresos"] / summary["Inversión"], np.nan)
    summary["Conv%"] = np.where(summary["Leads"] > 0, summary["Matriculados"] / summary["Leads"] * 100, np.nan)
    summary["% Alta Int."] = summary["Alta_Int"] * 100

    display = summary[[
        "ID_Campaña", "Canal", "Programa", "Inversión", "Leads", "CPL",
        "Entrevistas", "Coste_Ent", "Matriculados", "Ingresos", "ROAS", "Conv%", "% Alta Int.",
    ]].copy()
    display.columns = [
        "Campaña", "Canal", "Programa", "Inversión €", "Leads", "CPL €",
        "Entrevistas", "Coste Ent. €", "Matriculados", "Ingresos €", "ROAS", "Conv. %", "Alta Int. %",
    ]

    # Formateo para display
    for c in ["Inversión €", "CPL €", "Coste Ent. €", "Ingresos €"]:
        display[c] = display[c].apply(lambda x: f"{x:,.1f}" if not pd.isna(x) else "—")
    for c in ["ROAS"]:
        display[c] = display[c].apply(lambda x: f"{x:.2f}x" if not pd.isna(x) else "—")
    for c in ["Conv. %", "Alta Int. %"]:
        display[c] = display[c].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "—")

    st.dataframe(
        display.sort_values("Leads", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("CPL por Campaña")
        st.plotly_chart(chart_cpl_campanas(df), use_container_width=True, config={"displayModeBar": False})
    with c2:
        section("ROAS por Campaña")
        st.plotly_chart(chart_roas_campanas(df), use_container_width=True, config={"displayModeBar": False})

    section("Mapa de Eficiencia — CPL vs Leads (tamaño del círculo = Inversión)")
    st.caption("Las campañas ideales están en la zona inferior-derecha: bajo CPL y muchos leads.")
    st.plotly_chart(chart_mapa_eficiencia(df), use_container_width=True, config={"displayModeBar": False})

    section("% Leads Alta Intención por Semana y Canal")
    st.plotly_chart(chart_alta_intencion(df), use_container_width=True, config={"displayModeBar": False})


def tab_historico(df: pd.DataFrame):
    """Tab 3: Evolución histórica semana a semana."""
    section("Selecciona la métrica a analizar")
    metric = st.selectbox(
        "Métrica",
        ["Leads Válidos", "Inversión (€)", "CPL (€)", "Entrevistas", "Matriculados", "Ingresos (€)", "ROAS"],
        label_visibility="collapsed",
    )

    section(f"Evolución de {metric} por Campaña")
    fig = chart_evolucion_campana(df, metric)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No hay datos suficientes para esta métrica.")

    section("Heatmap — Rendimiento Semanal por Campaña")
    hm_metric = st.selectbox("Métrica heatmap", ["CPL (€)", "Leads Válidos", "Entrevistas", "ROAS"], key="hm")
    actual_col = hm_metric if hm_metric != "ROAS" else "ROAS"
    fig_hm = chart_heatmap_campanas(df, actual_col)
    if fig_hm:
        st.caption(
            "Verde = valor bajo (mejor para CPL) / Rojo = valor alto. "
            "Las celdas en blanco indican que la campaña no tenía datos esa semana."
        )
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

    section("Resumen histórico agregado por semana")
    hist = (
        df.groupby("Semana_label")
        .agg(
            Inversión=("Inversión (€)", "sum"),
            Leads=("Leads Válidos", "sum"),
            Entrevistas=("Entrevistas", "sum"),
            Matriculados=("Matriculados", "sum"),
            Ingresos=("Ingresos (€)", "sum"),
            Perdidos=("Perdidos", "sum"),
        )
        .reset_index()
    )
    hist["ROAS"] = np.where(hist["Inversión"] > 0, hist["Ingresos"] / hist["Inversión"], np.nan)
    hist["CPL"] = np.where(hist["Leads"] > 0, hist["Inversión"] / hist["Leads"], np.nan)
    hist["_ord"] = hist["Semana_label"].str.replace("S", "").astype(int)
    hist = hist.sort_values("_ord").drop(columns=["_ord"])
    hist["ROAS"] = hist["ROAS"].apply(lambda x: f"{x:.2f}x" if not pd.isna(x) else "—")
    hist["CPL"] = hist["CPL"].apply(lambda x: f"{x:.2f} €" if not pd.isna(x) else "—")
    hist["Inversión"] = hist["Inversión"].apply(lambda x: f"{x:,.2f} €")
    hist["Ingresos"] = hist["Ingresos"].apply(lambda x: f"{x:,.2f} €")
    st.dataframe(hist.rename(columns={"Semana_label": "Semana"}), use_container_width=True)


def tab_perdidas(df: pd.DataFrame):
    """Tab 4: Análisis de pérdidas y embudo de conversión."""
    c1, c2 = st.columns([1, 1])
    with c1:
        section("Embudo de Conversión Global")
        st.plotly_chart(chart_embudo(df), use_container_width=True, config={"displayModeBar": False})
    with c2:
        section("Motivos de Pérdida")
        fig_loss = chart_motivos_perdida(df)
        if fig_loss:
            st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Sin datos de motivos de pérdida para este filtro.")

    section("Pérdidas semanales vs Entrevistas vs Matrículas")
    st.plotly_chart(chart_perdida_por_semana(df), use_container_width=True, config={"displayModeBar": False})

    section("Análisis de pérdidas por campaña")
    loss_sum = df.groupby("ID_Campaña").agg(
        Leads=("Leads Válidos", "sum"),
        Perdidos=("Perdidos", "sum"),
        No_Valido=("No válido", "sum"),
        Precio=("No tiene dinero", "sum"),
        Competencia=("Matriculado en otra escuela", "sum"),
        Ilocalizado=("Ilocalizado", "sum"),
    ).reset_index()
    loss_sum["% Pérdida"] = np.where(
        loss_sum["Leads"] > 0,
        loss_sum["Perdidos"] / loss_sum["Leads"] * 100,
        np.nan,
    )
    loss_sum = loss_sum[loss_sum["Perdidos"] > 0].sort_values("% Pérdida", ascending=False)
    loss_sum["% Pérdida"] = loss_sum["% Pérdida"].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "—")
    st.dataframe(loss_sum.reset_index(drop=True), use_container_width=True, height=320)

    # Análisis orgánico separado
    org = df[df["Canal"].str.contains("orgánico|seo", case=False, na=False)]
    if not org.empty:
        section("Canal Orgánico — Métricas Clave")
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            kpi_card("🌱", "Leads Orgánicos", fmt_num(org["Leads Válidos"].sum()), "total")
        with oc2:
            kpi_card("🤝", "Entrevistas", fmt_num(org["Entrevistas"].sum()), "orgánico")
        with oc3:
            kpi_card("🎓", "Matriculados", fmt_num(org["Matriculados"].sum()), "orgánico")
        with oc4:
            kpi_card("💵", "Ingresos Org.", fmt_eur(org["Ingresos (€)"].sum()), "sin inversión directa")


def tab_datos(df: pd.DataFrame):
    """Tab 5: Tabla de datos raw con exportación."""
    section("Datos completos del período seleccionado")
    st.caption(f"Total filas: **{len(df):,}** · Semanas: {sorted(df['Semana'].unique().tolist())}")

    cols_show = [
        "Fecha de Análisis", "Semana_label", "ID_Campaña", "Canal", "Programa",
        "Inversión (€)", "Contactos", "Leads Válidos", "CPL (€)",
        "Entrevistas", "Coste Entrevista (€)", "Matriculados", "Ingresos (€)",
        "ROAS", "% Alta Intención", "% Lead→Entrevista", "% Entrevista→Matrícula",
        "Perdidos",
    ]
    cols_exist = [c for c in cols_show if c in df.columns]
    display = df[cols_exist].copy()
    display["Fecha de Análisis"] = display["Fecha de Análisis"].dt.strftime("%Y-%m-%d")
    display = display.rename(columns={"Semana_label": "Semana"})

    st.dataframe(display, use_container_width=True, height=500)

    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Exportar a CSV",
        data=csv,
        file_name="pontia_campanas_export.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    opts = get_filter_options(df)

    st.sidebar.markdown(
        """
        <div class="sb-brand">
            <span style="font-size:2rem">🧠</span>
            <div class="sb-title">PontIA</div>
            <div class="sb-sub">Marketing Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    email = st.session_state.get("user_email", "")
    st.sidebar.markdown(
        f'<div class="sb-user">👤 {email}</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 🔍 Filtros")

    semana = st.sidebar.selectbox("Semana", opts["semanas"], index=0)
    canal = st.sidebar.selectbox("Canal", opts["canales"], index=0)
    programa = st.sidebar.selectbox("Programa", opts["programas"], index=0)

    st.sidebar.markdown("---")

    # Estadísticas rápidas
    df_f = apply_filters(df, semana, canal, programa)
    st.sidebar.markdown(
        f"""
        <div style="font-size:.75rem;color:#8BA0B0;line-height:1.8">
        📊 <b style="color:#F6FAB2">{len(df_f):,}</b> registros filtrados<br>
        📅 <b style="color:#F6FAB2">{df_f['Semana'].nunique()}</b> semanas con datos<br>
        🎯 <b style="color:#F6FAB2">{df_f['ID_Campaña'].nunique()}</b> campañas activas
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Cerrar sesión"):
        st.session_state.clear()
        st.rerun()

    return df_f


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()

    # ── Auth ──────────────────────────────────────────────────────────────────
    if not st.session_state.get("authenticated"):
        show_login_page()
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Cargando datos…"):
        df_all = load_data()

    if df_all.empty:
        st.error("No se pudieron cargar los datos. Revisa la conexión al Google Sheet o el archivo Excel.")
        return

    # ── Sidebar + filtros ─────────────────────────────────────────────────────
    df_filtered = render_sidebar(df_all)

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown(
            "<h1 style='font-family:Manrope,sans-serif;font-weight:800;font-size:1.6rem;"
            "color:#F6FAB2;margin:0'>🧠 PontIA · Marketing Intelligence</h1>"
            "<p style='color:#8BA0B0;font-size:.85rem;margin:.15rem 0 .8rem'>Seguimiento semanal de campañas de captación</p>",
            unsafe_allow_html=True,
        )
    with h2:
        semanas_con_datos = sorted(df_all["Semana"].unique().tolist())
        ult = semanas_con_datos[-1] if semanas_con_datos else "—"
        st.markdown(
            f"<div style='text-align:right;font-family:IBM Plex Mono,monospace;font-size:.75rem;"
            f"color:#8BA0B0;padding-top:.5rem'>Última semana: <b style='color:#F6FAB2'>S{ult}</b><br>"
            f"Semanas totales: <b style='color:#F6FAB2'>{len(semanas_con_datos)}</b></div>",
            unsafe_allow_html=True,
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Resumen General",
        "🎯 Por Campaña",
        "📈 Evolución Histórica",
        "🚨 Análisis de Pérdidas",
        "📋 Datos",
    ])

    with t1:
        tab_resumen(df_filtered, df_all)
    with t2:
        tab_campanas(df_filtered)
    with t3:
        tab_historico(df_filtered)
    with t4:
        tab_perdidas(df_filtered)
    with t5:
        tab_datos(df_filtered)


if __name__ == "__main__":
    main()
