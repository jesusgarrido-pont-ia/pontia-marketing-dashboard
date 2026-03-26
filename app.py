"""
PontIA · Marketing Intelligence Dashboard
──────────────────────────────────────────
Cuadro de mandos semanal de campañas de marketing.
Conectado a Google Sheets (o Excel local como fallback).
"""

import base64
import calendar
import hashlib
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from utils.data_loader import apply_filters, get_filter_options, load_data

# ── AI module (optional) ────────────────────────────────────────────────────
try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


@st.cache_data
def _load_logo_b64() -> str:
    """Carga el isotipo de Pontia como base64 para uso inline en HTML."""
    logo_path = os.path.join(os.path.dirname(__file__), "Pontia_Logo_Isotipo_Black.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


# ── Benchmarks configurables ────────────────────────────────────────────────
_DEFAULT_BENCHMARKS = {
    "cpl": {"good": 15, "review": 25, "pause": 40, "optimal_line": 15},
    "roas": {"bad": 1, "good": 2, "excellent": 4},
    "coste_entrevista": {"good": 60, "bad": 100},
    "leads": {"good": 100, "bad": 50},
    "matriculados": {"good": 5, "bad": 2},
    "conv_lead_matricula": {"good": 5, "bad": 2},
}


@st.cache_data
def _load_benchmarks() -> dict:
    """Lee benchmarks de st.secrets primero, luego config.yaml, luego defaults."""
    benchmarks = dict(_DEFAULT_BENCHMARKS)
    # Try st.secrets first
    try:
        sb = dict(st.secrets.get("benchmarks", {}))
        if sb:
            for key in benchmarks:
                if key in sb:
                    benchmarks[key] = dict(sb[key])
            return benchmarks
    except Exception:
        pass
    # Try config.yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            sb = cfg.get("benchmarks", {})
            if sb:
                for key in benchmarks:
                    if key in sb:
                        benchmarks[key] = dict(sb[key])
        except Exception:
            pass
    return benchmarks

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PontIA · Marketing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# BRAND PALETTE
# ══════════════════════════════════════════════════════════════════════════════
C = {
    "bg":      "#F5F7FA",
    "bg2":     "#FFFFFF",
    "card":    "#FFFFFF",
    "green":   "#1B6B4A",
    "yellow":  "#EE7015",
    "amber":   "#D97706",
    "orange":  "#EE7015",
    "blue":    "#3B6FD4",
    "sage":    "#5BA88C",
    "red":     "#DC2626",
    "purple":  "#7C3AED",
    "ok":      "#16A34A",
    "warn":    "#F59E0B",
    "danger":  "#EF4444",
    "muted":   "#6B7280",
    "border":  "#E5E7EB",
}

CHANNEL_COLORS = {
    "Meta Ads (FB/IG)": C["orange"],
    "Google Ads":       C["blue"],
    "Orgánico / SEO":   C["sage"],
    "YouTube Ads":      C["amber"],
    "LinkedIn Ads":     C["purple"],
}

CHART_PALETTE = [
    C["blue"], C["orange"], C["amber"], C["sage"],
    C["purple"], "#00BCD4", C["ok"], "#FF80AB",
    "#FFB74D", "#81C784", "#EF5350", "#7C3AED",
]

LEGEND_BASE = dict(
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor=C["border"],
    borderwidth=1,
    font=dict(color="#1F2937", size=11),
)

AXIS_BASE = dict(gridcolor="#F0F0F0", linecolor=C["border"], tickfont=dict(color=C["muted"]), zerolinecolor=C["border"])

PLOT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Manrope, sans-serif", color="#1F2937", size=12),
    title_font=dict(family="Manrope, sans-serif", color="#1F2937", size=14),
    legend=LEGEND_BASE,
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=C["border"], font_family="Manrope", font_color="#1F2937"),
)

def _base(fig, title="", **axis_kwargs):
    """Aplica PLOT_BASE + ejes por defecto + overrides opcionales."""
    fig.update_layout(**PLOT_BASE, title_text=title, **axis_kwargs)
    if "xaxis" not in axis_kwargs:
        fig.update_xaxes(**AXIS_BASE)
    if "yaxis" not in axis_kwargs:
        fig.update_yaxes(**AXIS_BASE)
    return fig


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
html,body,[class*="css"]{font-family:'Manrope',-apple-system,sans-serif!important;color:#1F2937!important}
.stApp{background:#F5F7FA}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:1.2rem;padding-bottom:1rem;max-width:1500px}
a{color:#EE7015}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#FFFFFF 0%,#F9FAFB 100%)!important;border-right:1px solid #E5E7EB}
[data-testid="stSidebar"] .block-container{padding-top:.8rem}
[data-testid="stSidebar"] *{color:#374151!important}
[data-testid="stSidebar"] label{color:#374151!important}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown div{color:#374151!important}

/* ── Tabs ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{background:#FFFFFF;border-radius:10px;padding:4px;gap:4px;border:1px solid #E5E7EB;margin-bottom:.8rem}
.stTabs [data-baseweb="tab"]{background:transparent;color:#6B7280;border-radius:7px;padding:.45rem 1rem;font-family:'Manrope',sans-serif;font-weight:500;font-size:.875rem;transition:all .2s;border:none}
.stTabs [aria-selected="true"]{background:#EE7015!important;color:#FFFFFF!important;font-weight:700}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}

/* ── KPI Cards ────────────────────────────── */
.kpi-card{background:#FFFFFF;border:1px solid #E5E7EB;border-radius:14px;padding:1.1rem .9rem;text-align:center;transition:all .25s;position:relative;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.kpi-card::before{content:'';position:absolute;top:0;left:0;width:100%;height:3px;background:linear-gradient(90deg,#EE7015,#F59E0B)}
.kpi-card:hover{border-color:#EE7015;transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,.08)}
.kpi-icon{display:block;font-size:1.3rem;margin-bottom:.25rem}
.kpi-label{display:block;font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6B7280;margin-bottom:.35rem}
.kpi-value{display:block;font-family:'IBM Plex Mono',monospace;font-size:1.65rem;font-weight:600;color:#1F2937;line-height:1;margin-bottom:.25rem}
.kpi-sub{display:block;font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#9CA3AF}
.kpi-delta{display:block;font-size:.75rem;font-weight:700;margin-top:.2rem}
.badge{display:inline-block;font-size:.62rem;font-weight:700;padding:.12rem .45rem;border-radius:20px;margin-top:.25rem}
.bg{background:rgba(22,163,74,.1);color:#16A34A;border:1px solid rgba(22,163,74,.25)}
.by{background:rgba(245,158,11,.1);color:#D97706;border:1px solid rgba(245,158,11,.25)}
.br{background:rgba(239,68,68,.1);color:#EF4444;border:1px solid rgba(239,68,68,.25)}

/* ── Section title ────────────────────────── */
.sec{font-size:.8rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#EE7015;border-left:3px solid #EE7015;padding-left:.7rem;margin:1.2rem 0 .6rem}

/* ── Alert boxes ──────────────────────────── */
.al{border-radius:9px;padding:.75rem .9rem;font-size:.85rem;margin:.3rem 0;display:flex;align-items:flex-start;gap:.5rem;line-height:1.4}
.al-w{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.3);color:#92400E}
.al-d{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.3);color:#991B1B}
.al-s{background:rgba(22,163,74,.08);border:1px solid rgba(22,163,74,.3);color:#166534}
.al-i{background:rgba(59,111,212,.08);border:1px solid rgba(59,111,212,.3);color:#1E3A8A}

/* ── Buttons ──────────────────────────────── */
.stButton>button{background:linear-gradient(135deg,#EE7015,#F59E0B)!important;color:#FFFFFF!important;border:none!important;border-radius:8px!important;font-family:'Manrope',sans-serif!important;font-weight:700!important;font-size:.875rem!important;padding:.55rem 1.4rem!important;transition:all .2s!important;box-shadow:0 2px 8px rgba(238,112,21,.25)!important}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(238,112,21,.35)!important}

/* ── Inputs & Selects (FORCE light theme) ── */
.stSelectbox>div>div,
.stMultiSelect>div>div,
[data-baseweb="select"]>div{background:#FFFFFF!important;border-color:#E5E7EB!important;border-radius:8px!important;color:#1F2937!important}
[data-baseweb="select"]>div>div{color:#1F2937!important}
[data-baseweb="select"] svg{fill:#6B7280!important}
.stTextInput input,.stPasswordInput input{background:#FFFFFF!important;border-color:#E5E7EB!important;border-radius:8px!important;color:#1F2937!important}
.stMultiSelect [data-baseweb="tag"]{background:#FFF7ED!important;border:1px solid #FDBA74!important}
.stMultiSelect [data-baseweb="tag"] span{color:#9A3412!important;font-weight:600}

/* ── Dropdown/Popover menus (FORCE light) ── */
[data-baseweb="popover"]{background:#FFFFFF!important;border:1px solid #E5E7EB!important;border-radius:8px!important;box-shadow:0 4px 16px rgba(0,0,0,.1)!important}
[data-baseweb="popover"] *{color:#1F2937!important}
[data-baseweb="popover"] li{background:#FFFFFF!important;color:#1F2937!important}
[data-baseweb="popover"] li:hover{background:#FFF7ED!important}
[data-baseweb="popover"] [aria-selected="true"]{background:#FFF7ED!important}
[data-baseweb="menu"]{background:#FFFFFF!important}
[data-baseweb="list"]{background:#FFFFFF!important}
[data-baseweb="listbox"]{background:#FFFFFF!important}

/* ── Login ────────────────────────────────── */
.login-title{font-family:'Manrope',sans-serif;font-size:1.6rem;font-weight:800;color:#EE7015;text-align:center;margin-bottom:.2rem}
.login-sub{font-size:.85rem;color:#6B7280;text-align:center;margin-bottom:1.8rem}

/* ── Sidebar brand ────────────────────────── */
.sb-brand{text-align:center;padding:.5rem 0 1.2rem;border-bottom:1px solid #E5E7EB;margin-bottom:.8rem}
.sb-title{font-family:'Manrope',sans-serif;font-size:1.35rem;font-weight:800;color:#EE7015!important}
.sb-sub{font-size:.7rem;color:#6B7280!important;letter-spacing:.1em;text-transform:uppercase;margin-top:.1rem}
.sb-user{background:#FFF7ED;border:1px solid #FDBA74;border-radius:8px;padding:.5rem .7rem;font-size:.78rem;color:#9A3412!important;margin-bottom:.8rem}

/* ── Divider ──────────────────────────────── */
.div{height:1px;background:linear-gradient(90deg,#E5E7EB,transparent);margin:.8rem 0}

/* ── DataFrames ───────────────────────────── */
.stDataFrame{border-radius:10px;overflow:hidden}

/* ── Force ALL text to dark on light bg ──── */
.stMarkdown,.stMarkdown p,.stMarkdown span,.stMarkdown div,.stMarkdown li{color:#1F2937!important}
.stCaption,.stCaption *{color:#6B7280!important}

/* ── Expander ────────────────────────────── */
.streamlit-expanderHeader{background:#FFFFFF!important;color:#1F2937!important}
.streamlit-expanderContent{background:#FFFFFF!important}

/* ── Metric ──────────────────────────────── */
[data-testid="stMetric"]{background:#FFFFFF;border-radius:10px;padding:.8rem}
[data-testid="stMetricValue"]{color:#1F2937!important}
[data-testid="stMetricLabel"]{color:#6B7280!important}

/* ── KPI Grid (responsive) ───────────────── */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-bottom:.5rem}
.kpi-delta{font-family:'IBM Plex Mono',monospace;font-size:.7rem;font-weight:600;margin-top:.2rem}
.kpi-delta-up{color:#16A34A}
.kpi-delta-down{color:#EF4444}

/* ── Responsive ──────────────────────────── */
@media (max-width: 768px) {
  .kpi-grid{grid-template-columns:repeat(2,1fr)!important}
  .kpi-card{padding:.8rem .6rem}
  .kpi-value{font-size:1.3rem}
  .block-container{padding-left:.5rem;padding-right:.5rem}
}
</style>
""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════
def _load_auth_config() -> dict:
    """Lee credenciales de st.secrets o config.yaml (fallback).
    Soporta password_hash (SHA-256) y password (legacy)."""
    try:
        auth_sec = dict(st.secrets.get("auth", {}))
        if auth_sec:
            return auth_sec
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
    authorized = [e.lower().strip() for e in cfg.get("authorized_emails", []) if e]
    if email.lower().strip() not in authorized:
        return False
    # New hashed password mode
    if "password_hash" in cfg:
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        return input_hash == cfg["password_hash"]
    # Legacy plaintext password (backward compat)
    return password == cfg.get("password", "")


def show_login_page():
    inject_css()
    # Extra CSS para el contenedor de login
    st.markdown(
        """<style>
        .login-wrap [data-testid="column"]:nth-child(2) > div {
            background:#FFFFFF;border:1px solid #E5E7EB;border-radius:20px;
            padding:2.5rem 2rem;box-shadow:0 20px 60px rgba(0,0,0,.08);
        }
        </style>""",
        unsafe_allow_html=True,
    )
    logo_path = os.path.join(os.path.dirname(__file__), "Pontia_Logo_Isotipo_Black.png")
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        # Logo centrado
        lc1, lc2, lc3 = st.columns([1, 0.6, 1])
        with lc2:
            if os.path.exists(logo_path):
                st.image(logo_path, width=60)
        st.markdown(
            """
            <div style="text-align:center;margin-bottom:1.5rem;margin-top:-0.5rem">
                <span class="login-title">PontIA</span><br>
                <span class="login-sub">Marketing Intelligence &middot; Acceso privado</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Toggle entre login y recuperación de contraseña
        if st.session_state.get("show_forgot_pw"):
            st.markdown(
                '<p style="font-size:.85rem;color:#6B7280;text-align:center;margin-bottom:1rem">'
                'Introduce tu correo autorizado y te mostraremos la contraseña.</p>',
                unsafe_allow_html=True,
            )
            forgot_email = st.text_input("Correo electrónico", placeholder="tu@pontia.es", key="forgot_email")
            if st.button("Recuperar contraseña", use_container_width=True):
                st.info("Contacta con el administrador en admin@pontia.tech para recuperar tu contraseña.")
            st.markdown(
                '<div style="text-align:center;margin-top:.8rem">'
                '<span style="font-size:.82rem;color:#6B7280">¿Ya la recuerdas?</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("← Volver al inicio de sesión", use_container_width=True, type="secondary"):
                st.session_state["show_forgot_pw"] = False
                st.rerun()
        else:
            email = st.text_input("Correo electrónico", placeholder="tu@pontia.es", key="login_email")
            password = st.text_input("Contraseña", type="password", placeholder="••••••••", key="login_pw")
            if st.button("Entrar →", use_container_width=True):
                if check_login(email, password):
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"] = email
                    st.rerun()
                else:
                    st.error("Correo o contraseña incorrectos. Contacta con el administrador.")
            if st.button("¿Olvidaste tu contraseña?", use_container_width=True, type="secondary"):
                st.session_state["show_forgot_pw"] = True
                st.rerun()


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


def kpi_card(icon, label, value, sub="", badge_class="", badge_text="", delta=None, return_html=False):
    """Render a KPI card. If return_html=True, returns the HTML string instead of rendering."""
    badge_html = f'<span class="badge {badge_class}">{badge_text}</span>' if badge_class and badge_text else ""
    delta_html = ""
    if delta is not None and delta != "":
        if str(delta).startswith("-"):
            delta_cls = "kpi-delta-down"
        else:
            delta_cls = "kpi-delta-up"
        delta_html = f'<span class="kpi-delta {delta_cls}">{delta}</span>'
    html = (
        f'<div class="kpi-card">'
        f'<span class="kpi-icon">{icon}</span>'
        f'<span class="kpi-label">{label}</span>'
        f'<span class="kpi-value">{value}</span>'
        f'<span class="kpi-sub">{sub}</span>'
        f'{delta_html}'
        f'{badge_html}'
        f'</div>'
    )
    if return_html:
        return html
    st.markdown(html, unsafe_allow_html=True)


def kpi_grid(cards_html: list):
    """Render a responsive grid of KPI card HTML strings using st.columns."""
    # Use st.columns in rows of 4 for reliable rendering
    row_size = 4
    for i in range(0, len(cards_html), row_size):
        row = cards_html[i:i + row_size]
        cols = st.columns(len(row))
        for j, html in enumerate(row):
            with cols[j]:
                st.markdown(html, unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def alert(text, kind="i"):
    icons = {"w": "⚠️", "d": "🔴", "s": "✅", "i": "ℹ️"}
    st.markdown(f'<div class="al al-{kind}">{icons[kind]} {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
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
        ("Leads", C["amber"], "solid"),
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
    _base(fig, "Evolución Semanal — Inversión & Resultados")
    fig.update_layout(legend=dict(**LEGEND_BASE, orientation="h", y=-0.15))
    fig.update_yaxes(title_text="Inversión (€)", secondary_y=False,
                     title_font=dict(color=C["muted"]), tickfont=dict(color=C["muted"]),
                     gridcolor="#F0F0F0")
    fig.update_yaxes(title_text="Leads / Entrevistas / Matrículas", secondary_y=True,
                     title_font=dict(color=C["muted"]), tickfont=dict(color=C["muted"]),
                     gridcolor="#F0F0F0")
    return fig


def chart_roas_campanas(df: pd.DataFrame, benchmarks=None):
    """ROAS por campaña (barras horizontales, ordenadas)."""
    if benchmarks is None:
        benchmarks = _DEFAULT_BENCHMARKS
    b_roas = benchmarks.get("roas", _DEFAULT_BENCHMARKS["roas"])
    g = (
        df[df["Inversión (€)"].fillna(0) > 0]
        .groupby("ID_Campaña")
        .agg(Ingresos=("Ingresos (€)", "sum"), Inversión=("Inversión (€)", "sum"))
        .reset_index()
    )
    g["ROAS"] = g["Ingresos"] / g["Inversión"]
    g = g.sort_values("ROAS", ascending=True).tail(15)
    colors = [C["ok"] if r >= b_roas["excellent"] else C["warn"] if r >= b_roas["bad"] else C["danger"] for r in g["ROAS"]]
    fig = go.Figure(go.Bar(
        x=g["ROAS"], y=g["ID_Campaña"], orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>ROAS: %{x:.2f}x<extra></extra>",
        text=g["ROAS"].apply(lambda x: f"{x:.2f}x"),
        textposition="outside", textfont=dict(color="#1F2937", size=11),
    ))
    fig.add_vline(x=b_roas["bad"], line_dash="dash", line_color=C["warn"], annotation_text="Break-even",
                  annotation_font=dict(color=C["warn"], size=10))
    _base(fig, "ROAS por Campaña",
          xaxis=dict(**AXIS_BASE, title="ROAS"),
                      height=max(300, len(g) * 36 + 80))
    return fig


def chart_cpl_campanas(df: pd.DataFrame, benchmarks=None):
    """CPL y Leads Válidos por campaña (barras + scatter)."""
    if benchmarks is None:
        benchmarks = _DEFAULT_BENCHMARKS
    b_cpl = benchmarks.get("cpl", _DEFAULT_BENCHMARKS["cpl"])
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
    colors = [C["ok"] if v <= b_cpl["good"] else C["warn"] if v <= b_cpl["review"] else C["danger"] for v in g["CPL"]]
    fig = go.Figure(go.Bar(
        x=g["CPL"], y=g["ID_Campaña"], orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>CPL: %{x:.2f} €<br>Leads: %{customdata}<extra></extra>",
        customdata=g["Leads"],
        text=g["CPL"].apply(lambda x: f"{x:.1f} €"),
        textposition="outside", textfont=dict(color=C["muted"], size=10),
    ))
    fig.add_vline(x=b_cpl["optimal_line"], line_dash="dash", line_color=C["ok"],
                  annotation_text=f"Óptimo ≤{b_cpl['optimal_line']}€",
                  annotation_font=dict(color=C["ok"], size=10))
    _base(fig, "CPL Medio por Campaña (€)",
          xaxis=dict(**AXIS_BASE, title="CPL (€)"),
                      height=max(300, len(g) * 36 + 80))
    return fig


@st.cache_data
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
    _base(fig, "Distribución por Canal",
                      showlegend=False, height=320)
    return fig


@st.cache_data
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
    _base(fig, "Mapa de Eficiencia — CPL vs Leads (tamaño = Inversión)",
          xaxis=dict(**AXIS_BASE, title="CPL (€) — menor es mejor"),
          yaxis=dict(**AXIS_BASE, title="Leads Válidos"),
                      height=480)
    return fig


@st.cache_data
def chart_alta_intencion(df: pd.DataFrame):
    """% Alta Intención por semana y canal."""
    # Usar % Alta Intención precalculada si está disponible, si no calcular desde Consideración/Decisión
    if "% Alta Intención" in df.columns:
        g = (
            df.groupby(["Semana_label", "Canal"])
            .agg(Leads=("Leads Válidos", "sum"), AltaInt=("% Alta Intención", "mean"))
            .reset_index()
        )
        g = g[g["Leads"] > 0]
        g["% Alta Int."] = g["AltaInt"].fillna(0) * 100
    elif "Consideración" in df.columns and "Decisión" in df.columns:
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
    else:
        return None
    g["_ord"] = g["Semana_label"].str.replace("S", "").astype(int)
    g = g.sort_values("_ord")

    fig = px.line(
        g, x="Semana_label", y="% Alta Int.", color="Canal",
        color_discrete_map=CHANNEL_COLORS,
        markers=True, line_shape="spline",
        hover_data={"Leads": True},
    )
    fig.update_traces(line_width=2.5, marker_size=7)
    _base(fig, "% Leads de Alta Intención por Semana y Canal",
          yaxis=dict(**AXIS_BASE, title="% Alta Intención", ticksuffix="%"))
    fig.update_layout(legend=dict(**LEGEND_BASE, orientation="h", y=-0.2))
    return fig


@st.cache_data
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
        marker_color=[C["blue"], C["amber"], C["orange"], C["ok"]],
        connector=dict(line=dict(color=C["border"], width=1)),
        hovertemplate="<b>%{y}</b><br>Total: %{x}<br>Del total: %{percentInitial}<extra></extra>",
    ))
    _base(fig, "Embudo de Conversión Global", height=320)
    return fig


@st.cache_data
def chart_motivos_perdida(df: pd.DataFrame):
    """Desglose de motivos de pérdida (donut)."""
    loss_cols = {
        "No válido":               "No válido",
        "No es lo que buscaba":    "No es lo que buscaba",
        "No tiene dinero":         "Precio",
        "No interesa (Otros)":     "No interesa",
        "Matriculado en otra escuela": "Competencia",
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
    _base(fig, "Motivos de Pérdida de Leads", height=350, showlegend=False)
    return fig


def chart_heatmap_campanas(df: pd.DataFrame, metric="CPL (€)"):
    """Heatmap: Campañas × Semanas con color = métrica."""
    if metric not in df.columns:
        return None
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
    _base(fig, f"Heatmap — {metric} por Campaña × Semana",
          yaxis={**AXIS_BASE, "tickfont": dict(size=10, color=C["muted"])},
          height=max(300, len(pivot) * 30 + 100))
    return fig


def chart_programa_canal(df: pd.DataFrame, benchmarks=None, metric="CPL (€)"):
    """Grouped bar chart: X=Programa, color=Canal, Y=selected metric."""
    if benchmarks is None:
        benchmarks = _DEFAULT_BENCHMARKS
    agg_func = "mean" if metric in ("CPL (€)", "ROAS") else "sum"
    g = df.groupby(["Programa", "Canal"]).agg(val=(metric, agg_func)).reset_index()
    if g.empty:
        return None
    fig = px.bar(
        g, x="Programa", y="val", color="Canal",
        barmode="group",
        color_discrete_map=CHANNEL_COLORS,
        labels={"val": metric, "Programa": "Programa"},
    )
    _base(fig, f"{metric} por Programa y Canal", height=420)
    fig.update_layout(legend=dict(**LEGEND_BASE, orientation="h", y=-0.2))
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
    _base(fig, f"Evolución de {metric} por Campaña", height=420)
    fig.update_layout(legend=dict(**LEGEND_BASE, orientation="h", y=-0.2))
    return fig


@st.cache_data
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
    _base(fig, "Entrevistas, Matrículas y Pérdidas por Semana",
          barmode="group")
    fig.update_layout(legend=dict(**LEGEND_BASE, orientation="h", y=-0.15))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SEMÁFORO DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
def _clasificar_campana(cpl, roas, leads, inv, benchmarks=None):
    """Devuelve (icono, estado, color_borde, color_fondo, color_texto, color_nombre)."""
    if benchmarks is None:
        benchmarks = _DEFAULT_BENCHMARKS
    b_cpl = benchmarks.get("cpl", _DEFAULT_BENCHMARKS["cpl"])
    b_roas = benchmarks.get("roas", _DEFAULT_BENCHMARKS["roas"])

    if leads == 0 and inv > 0:
        return ("🔴", "PAUSAR",
                "rgba(239,68,68,.3)", "rgba(239,68,68,.06)", "#DC2626", "#991B1B")
    if pd.isna(cpl) or cpl is None:
        return ("⚪", "S/D", "rgba(107,114,128,.25)", "rgba(107,114,128,.05)", "#6B7280", "#6B7280")
    if cpl > b_cpl["pause"]:
        return ("🔴", "PAUSAR",
                "rgba(239,68,68,.3)", "rgba(239,68,68,.06)", "#DC2626", "#991B1B")
    if cpl > b_cpl["review"] or (not pd.isna(roas) and roas is not None and roas < b_roas["bad"]):
        return ("🟡", "REVISAR",
                "rgba(245,158,11,.3)", "rgba(245,158,11,.06)", "#D97706", "#92400E")
    if cpl <= b_cpl["good"] and (pd.isna(roas) or roas is None or roas >= b_roas["good"]):
        return ("🟢", "ESCALAR",
                "rgba(22,163,74,.3)", "rgba(22,163,74,.06)", "#16A34A", "#166534")
    return ("⚪", "MANTENER",
            "rgba(59,111,212,.25)", "rgba(59,111,212,.05)", "#3B6FD4", "#1E3A8A")


def panel_decisiones(df: pd.DataFrame, benchmarks=None):
    """Panel de decisiones con semáforo — para la reunión semanal."""
    g = (
        df.groupby("ID_Campaña")
        .agg(
            Canal=("Canal", "first"),
            Leads=("Leads Válidos", "sum"),
            Inv=("Inversión (€)", "sum"),
            CPL=("CPL (€)", "mean"),
            ROAS=("ROAS", "mean"),
            Matr=("Matriculados", "sum"),
        )
        .reset_index()
    )
    estados = g.apply(
        lambda r: pd.Series(
            _clasificar_campana(r["CPL"], r["ROAS"], r["Leads"], r["Inv"], benchmarks),
            index=["ico", "estado", "borde", "fondo", "texto", "nombre_color"],
        ),
        axis=1,
    )
    g = pd.concat([g, estados], axis=1)

    pausar  = g[g["estado"] == "PAUSAR"].sort_values("CPL", ascending=False, na_position="first")
    revisar = g[g["estado"] == "REVISAR"].sort_values("CPL", ascending=False)
    escalar = g[g["estado"] == "ESCALAR"].sort_values("ROAS", ascending=False)

    # ── Cabecera ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:#FFFFFF;
        border:1px solid #E5E7EB;border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1.2rem;box-shadow:0 1px 3px rgba(0,0,0,.06)">
          <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.9rem">
            <span style="font-size:1.2rem">🎯</span>
            <span style="font-family:Manrope,sans-serif;font-size:1rem;font-weight:800;color:#1F2937">
              Panel de Decisiones — ¿Qué hacer esta semana?
            </span>
            <span style="font-size:.72rem;color:#6B7280;margin-left:auto">{len(g)} campañas</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem">
            <div style="background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.2);
            border-radius:10px;padding:.7rem;text-align:center">
              <div style="font-size:1.8rem;font-weight:800;color:#DC2626">{len(pausar)}</div>
              <div style="font-size:.68rem;font-weight:700;color:#991B1B;letter-spacing:.08em">🔴 PAUSAR</div>
            </div>
            <div style="background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);
            border-radius:10px;padding:.7rem;text-align:center">
              <div style="font-size:1.8rem;font-weight:800;color:#D97706">{len(revisar)}</div>
              <div style="font-size:.68rem;font-weight:700;color:#92400E;letter-spacing:.08em">🟡 REVISAR</div>
            </div>
            <div style="background:rgba(22,163,74,.06);border:1px solid rgba(22,163,74,.2);
            border-radius:10px;padding:.7rem;text-align:center">
              <div style="font-size:1.8rem;font-weight:800;color:#16A34A">{len(escalar)}</div>
              <div style="font-size:.68rem;font-weight:700;color:#166534;letter-spacing:.08em">🟢 ESCALAR</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tres columnas con tarjetas ─────────────────────────────────────────
    def _tarjeta(r):
        cpl_s = f"{r['CPL']:.1f} €" if not pd.isna(r["CPL"]) else "Sin CPL"
        roas_s = f"· ROAS {r['ROAS']:.2f}x" if not pd.isna(r["ROAS"]) else ""
        leads_s = int(r["Leads"])
        return (
            f'<div style="background:{r["fondo"]};border:1px solid {r["borde"]};'
            f'border-radius:9px;padding:.55rem .8rem;margin-bottom:.35rem">'
            f'<div style="font-weight:700;font-size:.82rem;color:{r["nombre_color"]};'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{r["ID_Campaña"]}</div>'
            f'<div style="font-size:.7rem;color:#6B7280;margin-top:.1rem">'
            f'CPL: <b style="color:{r["texto"]}">{cpl_s}</b> · Leads: <b>{leads_s}</b>{roas_s}'
            f"</div></div>"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div style="font-size:.8rem;font-weight:700;color:#DC2626;margin-bottom:.5rem">'
            "🔴 Pausar — CPL alto o sin leads</div>",
            unsafe_allow_html=True,
        )
        if pausar.empty:
            st.markdown(
                '<div style="font-size:.78rem;color:#6B7280;font-style:italic">Ninguna campaña en zona roja ✓</div>',
                unsafe_allow_html=True,
            )
        for _, r in pausar.iterrows():
            st.markdown(_tarjeta(r), unsafe_allow_html=True)

    with c2:
        st.markdown(
            '<div style="font-size:.8rem;font-weight:700;color:#D97706;margin-bottom:.5rem">'
            "🟡 Revisar — CPL elevado o ROAS < 1</div>",
            unsafe_allow_html=True,
        )
        if revisar.empty:
            st.markdown(
                '<div style="font-size:.78rem;color:#6B7280;font-style:italic">Ninguna en zona amarilla ✓</div>',
                unsafe_allow_html=True,
            )
        for _, r in revisar.iterrows():
            st.markdown(_tarjeta(r), unsafe_allow_html=True)

    with c3:
        st.markdown(
            '<div style="font-size:.8rem;font-weight:700;color:#16A34A;margin-bottom:.5rem">'
            "🟢 Escalar — Bajo CPL y buen retorno</div>",
            unsafe_allow_html=True,
        )
        if escalar.empty:
            st.markdown(
                '<div style="font-size:.78rem;color:#6B7280;font-style:italic">Ninguna lista para escalar aún</div>',
                unsafe_allow_html=True,
            )
        for _, r in escalar.iterrows():
            st.markdown(_tarjeta(r), unsafe_allow_html=True)

    return g  # Para reutilizar en la tabla


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
def _get_previous_period_data(df: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    """Returns data from the previous period based on current filters.
    If a single week is selected, returns previous week.
    If multiple weeks, returns same-length previous period."""
    current_weeks = sorted(df["Semana"].unique().tolist())
    if not current_weeks:
        return pd.DataFrame()
    all_weeks = sorted(df_all["Semana"].unique().tolist())
    n = len(current_weeks)
    min_week = min(current_weeks)
    prev_weeks = [w for w in all_weeks if w < min_week]
    if not prev_weeks:
        return pd.DataFrame()
    prev_weeks = prev_weeks[-n:]  # take same number of weeks
    return df_all[df_all["Semana"].isin(prev_weeks)]


def generate_ai_summary(kpis: dict, benchmarks: dict) -> str:
    """Genera un resumen con IA basado en los KPIs actuales.
    Requiere API key de Anthropic en st.secrets."""
    api_key = ""
    try:
        api_key = st.secrets.get("anthropic", {}).get("api_key", "")
    except Exception:
        pass
    if not api_key or not _HAS_ANTHROPIC:
        return ""

    prompt = (
        f"Eres un analista de marketing digital experto. Analiza estos KPIs de campañas de captación "
        f"de una escuela de formación online y genera un resumen ejecutivo en español (3-5 puntos clave, "
        f"máx 200 palabras). Incluye recomendaciones accionables.\n\n"
        f"KPIs del período:\n"
        f"- Inversión total: {kpis.get('inv', 0):,.0f} €\n"
        f"- Leads válidos: {kpis.get('leads', 0)}\n"
        f"- CPL medio: {kpis.get('cpl', 0):.2f} € (benchmark: ≤{benchmarks['cpl']['good']}€)\n"
        f"- Entrevistas: {kpis.get('entrevistas', 0)}\n"
        f"- Matriculados: {kpis.get('matriculados', 0)}\n"
        f"- Ingresos: {kpis.get('ingresos', 0):,.0f} €\n"
        f"- ROAS: {kpis.get('roas', 0):.2f}x (benchmark: ≥{benchmarks['roas']['good']}x)\n"
        f"- Conv. Lead→Matrícula: {kpis.get('conv_mat', 0):.1%}\n"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"Error al generar resumen: {e}"


def tab_resumen(df: pd.DataFrame, df_all: pd.DataFrame, benchmarks=None):
    """Tab 1: Resumen general con KPIs y gráficas clave."""
    if benchmarks is None:
        benchmarks = _load_benchmarks()
    b_cpl = benchmarks.get("cpl", _DEFAULT_BENCHMARKS["cpl"])
    b_roas = benchmarks.get("roas", _DEFAULT_BENCHMARKS["roas"])
    b_ce = benchmarks.get("coste_entrevista", _DEFAULT_BENCHMARKS["coste_entrevista"])
    b_leads = benchmarks.get("leads", _DEFAULT_BENCHMARKS["leads"])
    b_mat = benchmarks.get("matriculados", _DEFAULT_BENCHMARKS["matriculados"])

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

    # ── WoW deltas ───────────────────────────────────────────────────────────
    prev = _get_previous_period_data(df, df_all)
    deltas = {}
    if not prev.empty:
        p_inv = prev["Inversión (€)"].sum()
        p_leads = int(prev["Leads Válidos"].sum())
        p_cpl = prev["CPL (€)"].mean()
        p_entrevistas = int(prev["Entrevistas"].sum())
        p_coste_ent = p_inv / p_entrevistas if p_entrevistas > 0 else None
        p_matriculados = int(prev["Matriculados"].sum())
        p_ingresos = prev["Ingresos (€)"].sum()
        p_roas = p_ingresos / p_inv if p_inv > 0 else None

        def _delta_pct(curr, prev_val):
            if prev_val and prev_val != 0 and not pd.isna(prev_val) and curr is not None and not pd.isna(curr):
                d = (curr - prev_val) / abs(prev_val) * 100
                sign = "+" if d >= 0 else ""
                return f"{sign}{d:.1f}%"
            return None

        deltas["inv"] = _delta_pct(inv, p_inv)
        deltas["leads"] = _delta_pct(leads, p_leads)
        deltas["cpl"] = _delta_pct(cpl, p_cpl)
        deltas["entrevistas"] = _delta_pct(entrevistas, p_entrevistas)
        deltas["coste_ent"] = _delta_pct(coste_ent, p_coste_ent)
        deltas["matriculados"] = _delta_pct(matriculados, p_matriculados)
        deltas["ingresos"] = _delta_pct(ingresos, p_ingresos)
        deltas["roas"] = _delta_pct(roas, p_roas)

    # ── Budget projection ────────────────────────────────────────────────────
    budget_proj = None
    try:
        dates = df["Fecha de Análisis"].dropna()
        if not dates.empty:
            max_date = dates.max()
            min_date = dates.min()
            days_elapsed = max((max_date - min_date).days + 1, 1)
            year_m, month_m = max_date.year, max_date.month
            days_in_month = calendar.monthrange(year_m, month_m)[1]
            budget_proj = inv / days_elapsed * days_in_month
    except Exception:
        pass

    # ── KPI cards as responsive grid ─────────────────────────────────────────
    cards = []
    cards.append(kpi_card("💰", "Inversión Total", fmt_eur(inv), "acumulado",
                          delta=deltas.get("inv"), return_html=True))
    cards.append(kpi_card("📥", "Leads Válidos", fmt_num(leads), "total",
                          _color_badge(leads, b_leads["good"], b_leads["bad"]),
                          f"{'▲' if leads > b_leads['good'] else '▼'} {leads}",
                          delta=deltas.get("leads"), return_html=True))
    badge = _color_badge(cpl, b_cpl["good"], b_cpl["review"], invert=True) if cpl else ""
    cards.append(kpi_card("💡", "CPL Medio", fmt_eur(cpl, 2), "coste por lead", badge,
                          f"≤{b_cpl['good']}€ óptimo" if badge == "bg" else f"&gt;{b_cpl['review']}€ alto" if badge == "br" else "aceptable",
                          delta=deltas.get("cpl"), return_html=True))
    cards.append(kpi_card("🤝", "Entrevistas", fmt_num(entrevistas), "total",
                          delta=deltas.get("entrevistas"), return_html=True))
    badge = _color_badge(coste_ent, b_ce["good"], b_ce["bad"], invert=True) if coste_ent else ""
    cards.append(kpi_card("💼", "Coste/Entrevista", fmt_eur(coste_ent, 1) if coste_ent else "—", "€ por entrevista",
                          badge, "", delta=deltas.get("coste_ent"), return_html=True))
    cards.append(kpi_card("🎓", "Matriculados", fmt_num(matriculados), "total",
                          _color_badge(matriculados, b_mat["good"], b_mat["bad"]), "",
                          delta=deltas.get("matriculados"), return_html=True))
    cards.append(kpi_card("💵", "Ingresos", fmt_eur(ingresos), "acumulado",
                          delta=deltas.get("ingresos"), return_html=True))
    badge = _color_badge(roas, 3, b_roas["bad"]) if roas else ""
    cards.append(kpi_card("📈", "ROAS Global", f"{roas:.2f}x" if roas else "—", "retorno inversión",
                          badge, f"≥3x bueno" if badge == "bg" else f"&lt;{b_roas['bad']}x negativo" if badge == "br" else "",
                          delta=deltas.get("roas"), return_html=True))
    # Budget projection card
    if budget_proj is not None:
        cards.append(kpi_card("📅", "Proyección Mes", fmt_eur(budget_proj), "gasto estimado mensual",
                              return_html=True))

    kpi_grid(cards)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── AI Summary (Phase 5) ─────────────────────────────────────────────
    with st.expander("🤖 Resumen con IA (beta)"):
        api_key_set = False
        try:
            api_key_set = bool(st.secrets.get("anthropic", {}).get("api_key", ""))
        except Exception:
            pass
        if not _HAS_ANTHROPIC:
            st.info("El módulo `anthropic` no está instalado. Instálalo con `pip install anthropic>=0.40.0`.")
        elif not api_key_set:
            st.info("Añade tu API key de Anthropic en Secrets (anthropic.api_key) para activar el resumen con IA.")
        else:
            if st.button("Generar resumen con IA", key="ai_summary_btn"):
                with st.spinner("Generando resumen..."):
                    kpis_dict = {
                        "inv": inv, "leads": leads, "cpl": cpl if not pd.isna(cpl) else 0,
                        "entrevistas": entrevistas, "matriculados": matriculados,
                        "ingresos": ingresos, "roas": roas if roas else 0,
                        "conv_mat": conv_mat if conv_mat else 0,
                    }
                    summary_text = generate_ai_summary(kpis_dict, benchmarks)
                    if summary_text:
                        st.markdown(summary_text)
                    else:
                        st.warning("No se pudo generar el resumen.")

    # ── Panel de Decisiones ────────────────────────────────────────────────
    section("Decisiones de la Semana")
    panel_decisiones(df, benchmarks)

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
            alert(f"Tasa de conversión Lead→Matrícula: <b>{fmt_pct(conv_mat)}</b> — benchmark sector ~3-5%.", "i")

    # ── Gráficas ───────────────────────────────────────────────────────────
    section("Evolución Semanal")
    fig_evo = chart_evolucion_semanal(df_all)
    st.plotly_chart(fig_evo, use_container_width=True, config={"displayModeBar": False}, key="evo_semanal")

    c1, c2 = st.columns(2)
    with c1:
        section("ROAS por Campaña")
        st.plotly_chart(chart_roas_campanas(df, benchmarks), use_container_width=True, config={"displayModeBar": False}, key="roas_resumen")
    with c2:
        section("Distribución por Canal")
        st.plotly_chart(chart_distribucion_canal(df), use_container_width=True, config={"displayModeBar": False}, key="dist_canal_resumen")


def tab_campanas(df: pd.DataFrame, benchmarks=None):
    """Tab 2: Análisis detallado por campaña."""
    if benchmarks is None:
        benchmarks = _load_benchmarks()
    b_cpl = benchmarks.get("cpl", _DEFAULT_BENCHMARKS["cpl"])
    b_roas = benchmarks.get("roas", _DEFAULT_BENCHMARKS["roas"])
    b_conv = benchmarks.get("conv_lead_matricula", _DEFAULT_BENCHMARKS["conv_lead_matricula"])

    section("Tabla de Rendimiento por Campaña")
    st.caption("Los colores de CPL, ROAS y Conv. indican el rendimiento: 🟢 bueno · 🟡 mejorable · 🔴 crítico")

    summary = (
        df.groupby(["ID_Campaña", "Canal", "Programa"])
        .agg(
            Inversión=("Inversión (€)", "sum"),
            Leads=("Leads Válidos", "sum"),
            CPL=("CPL (€)", "mean"),
            Entrevistas=("Entrevistas", "sum"),
            Matriculados=("Matriculados", "sum"),
            Ingresos=("Ingresos (€)", "sum"),
            Alta_Int=("% Alta Intención", "mean"),
        )
        .reset_index()
    )
    summary["ROAS"] = np.where(summary["Inversión"] > 0, summary["Ingresos"] / summary["Inversión"], np.nan)
    summary["Conv%"] = np.where(summary["Leads"] > 0, summary["Matriculados"] / summary["Leads"] * 100, np.nan)

    # Columna Estado con semáforo
    summary["Estado"] = summary.apply(
        lambda r: _clasificar_campana(r["CPL"], r["ROAS"], r["Leads"], r["Inversión"], benchmarks)[0]
                  + " " + _clasificar_campana(r["CPL"], r["ROAS"], r["Leads"], r["Inversión"], benchmarks)[1],
        axis=1,
    )

    display = summary[[
        "Estado", "ID_Campaña", "Canal", "Inversión", "Leads", "CPL",
        "Entrevistas", "Matriculados", "Ingresos", "ROAS", "Conv%",
    ]].copy()
    display.columns = [
        "Estado", "Campaña", "Canal", "Inversión €", "Leads", "CPL €",
        "Entrevistas", "Matrículas", "Ingresos €", "ROAS", "Conv. %",
    ]

    # Funciones de color para Styler usando benchmarks
    cpl_good = b_cpl["good"]
    cpl_review = b_cpl["review"]
    roas_good_t = b_roas["good"]
    roas_bad_t = b_roas["bad"]
    conv_good_t = b_conv["good"]
    conv_bad_t = b_conv["bad"]

    def _c_cpl(v):
        if pd.isna(v): return ""
        if v <= cpl_good: return "color: #4CAF50; font-weight: 700"
        if v <= cpl_review: return "color: #FFC107; font-weight: 700"
        return "color: #EF5350; font-weight: 700"

    def _c_roas(v):
        if pd.isna(v): return ""
        if v >= roas_good_t: return "color: #4CAF50; font-weight: 700"
        if v >= roas_bad_t: return "color: #FFC107; font-weight: 700"
        return "color: #EF5350; font-weight: 700"

    def _c_conv(v):
        if pd.isna(v): return ""
        if v >= conv_good_t: return "color: #4CAF50; font-weight: 700"
        if v >= conv_bad_t: return "color: #FFC107; font-weight: 700"
        return "color: #EF5350; font-weight: 700"

    styled = (
        display.sort_values("Leads", ascending=False)
        .reset_index(drop=True)
        .style
        .format({
            "Inversión €": lambda x: f"{x:,.0f} €" if not pd.isna(x) else "—",
            "Leads": lambda x: f"{int(x):,}" if not pd.isna(x) else "—",
            "CPL €": lambda x: f"{x:.1f} €" if not pd.isna(x) else "—",
            "Entrevistas": lambda x: f"{int(x):,}" if not pd.isna(x) else "—",
            "Matrículas": lambda x: f"{int(x):,}" if not pd.isna(x) else "—",
            "Ingresos €": lambda x: f"{x:,.0f} €" if not pd.isna(x) else "—",
            "ROAS": lambda x: f"{x:.2f}x" if not pd.isna(x) else "—",
            "Conv. %": lambda x: f"{x:.1f}%" if not pd.isna(x) else "—",
        })
        .applymap(_c_cpl, subset=["CPL €"])
        .applymap(_c_roas, subset=["ROAS"])
        .applymap(_c_conv, subset=["Conv. %"])
    )

    st.dataframe(styled, use_container_width=True, height=420)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section("CPL por Campaña")
        st.plotly_chart(chart_cpl_campanas(df, benchmarks), use_container_width=True, config={"displayModeBar": False}, key="cpl_campanas")
    with c2:
        section("ROAS por Campaña")
        st.plotly_chart(chart_roas_campanas(df, benchmarks), use_container_width=True, config={"displayModeBar": False}, key="roas_campanas")

    section("Mapa de Eficiencia — CPL vs Leads (tamaño del círculo = Inversión)")
    st.caption("Las campañas ideales están en la zona inferior-derecha: bajo CPL y muchos leads.")
    st.plotly_chart(chart_mapa_eficiencia(df), use_container_width=True, config={"displayModeBar": False}, key="mapa_eficiencia")

    # ── Program Attribution (Phase 4) ────────────────────────────────────────
    section("Rendimiento por Programa y Canal")
    prog_metric = st.selectbox(
        "Métrica", ["CPL (€)", "ROAS", "Leads Válidos"],
        key="prog_canal_metric",
    )
    fig_prog = chart_programa_canal(df, benchmarks, prog_metric)
    if fig_prog:
        st.plotly_chart(fig_prog, use_container_width=True, config={"displayModeBar": False}, key="prog_canal")
    else:
        st.info("Sin datos de programa/canal para este filtro.")

    section("% Leads Alta Intención por Semana y Canal")
    fig_ai = chart_alta_intencion(df)
    if fig_ai:
        st.plotly_chart(fig_ai, use_container_width=True, config={"displayModeBar": False}, key="alta_intencion")
    else:
        st.info("Sin datos de intención disponibles.")


def _chart_evolucion_metric(agg: pd.DataFrame, col: str, title: str, campaigns: list, fmt: str = "") -> go.Figure:
    """Gráfica de líneas: evolución semanal de una métrica para las campañas seleccionadas."""
    fig = go.Figure()
    for i, camp in enumerate(campaigns):
        d = agg[agg["ID_Campaña"] == camp].sort_values("Semana")
        if d[col].dropna().empty:
            continue
        color = CHART_PALETTE[i % len(CHART_PALETTE)]
        fig.add_trace(go.Scatter(
            x=d["Semana_label"], y=d[col],
            name=camp[:22], mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
            connectgaps=True,
            hovertemplate=f"<b>{camp}</b><br>%{{x}}<br>{title}: %{{y:.1f}}{fmt}<extra></extra>",
        ))
    _base(fig, title, height=320)
    fig.update_layout(legend={**LEGEND_BASE, "orientation": "h", "y": -0.32, "font": dict(size=9)})
    return fig


def tab_historico(df: pd.DataFrame):
    """Tab 3: Evolución de campañas en el tiempo — decisiones de presupuesto."""
    section("Evolución de Campañas en el Tiempo")

    # ── Selector de campañas ──────────────────────────────────────────────────
    all_camps = sorted(df["ID_Campaña"].unique().tolist())
    top5 = (df.groupby("ID_Campaña")["Leads Válidos"].sum().nlargest(5).index.tolist())
    default = [c for c in top5 if c in all_camps]

    selected = st.multiselect(
        "Selecciona las campañas a comparar",
        options=all_camps,
        default=default,
        help="Por defecto se muestran las 5 campañas con más leads.",
    )
    if not selected:
        st.warning("Selecciona al menos una campaña para ver la evolución.")
        return

    # ── Agregar por semana + campaña ─────────────────────────────────────────
    dfs = df[df["ID_Campaña"].isin(selected)].copy()
    agg = (
        dfs.groupby(["Semana_label", "Semana", "ID_Campaña"])
        .agg(
            Inversión=("Inversión (€)", "sum"),
            Leads=("Leads Válidos", "sum"),
            Entrevistas=("Entrevistas", "sum"),
            Matriculados=("Matriculados", "sum"),
            Ingresos=("Ingresos (€)", "sum"),
            CosteEnt=("Coste Entrevista (€)", "mean"),
        )
        .reset_index()
    )
    agg["CPL"] = np.where(agg["Leads"] > 0, agg["Inversión"] / agg["Leads"], np.nan)
    agg["ROAS"] = np.where(agg["Inversión"] > 0, agg["Ingresos"] / agg["Inversión"], np.nan)

    st.markdown("---")

    # ── 6 gráficas en grid 2×3 ───────────────────────────────────────────────
    metrics = [
        ("CPL",         "CPL — Coste por Lead (€)",         "€"),
        ("ROAS",        "ROAS — Retorno sobre Inversión",    "x"),
        ("Leads",       "Leads Válidos",                     ""),
        ("Entrevistas", "Entrevistas realizadas",            ""),
        ("CosteEnt",    "Coste por Entrevista (€)",          "€"),
        ("Ingresos",    "Ingresos generados (€)",            "€"),
    ]

    for i in range(0, len(metrics), 2):
        c1, c2 = st.columns(2)
        for col_ui, (col_data, title, fmt) in zip([c1, c2], metrics[i:i+2]):
            with col_ui:
                fig = _chart_evolucion_metric(agg, col_data, title, selected, fmt)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"evol_{col_data}")

    # ── Tabla de tendencias ───────────────────────────────────────────────────
    st.markdown("---")
    section("Tabla de tendencias — última semana vs semana anterior")
    st.caption("🟢 Mejora · 🔴 Empeora · ⚪ Sin cambio o datos insuficientes")

    rows = []
    for camp in selected:
        d = agg[agg["ID_Campaña"] == camp].sort_values("Semana")
        if len(d) < 2:
            continue
        last, prev = d.iloc[-1], d.iloc[-2]

        def trend(new, old, lower_better=False):
            if pd.isna(new) or pd.isna(old) or old == 0:
                return "⚪"
            better = new < old if lower_better else new > old
            return "🟢" if better else "🔴"

        rows.append({
            "Campaña": camp,
            f"CPL S{int(last.Semana)}": f"{last.CPL:.1f} €" if not pd.isna(last.CPL) else "—",
            "CPL ↗": trend(last.CPL, prev.CPL, lower_better=True),
            f"ROAS S{int(last.Semana)}": f"{last.ROAS:.2f}x" if not pd.isna(last.ROAS) else "—",
            "ROAS ↗": trend(last.ROAS, prev.ROAS),
            f"Leads S{int(last.Semana)}": int(last.Leads),
            "Leads ↗": trend(last.Leads, prev.Leads),
            f"Ingresos S{int(last.Semana)}": f"{last.Ingresos:,.0f} €",
            "Ingresos ↗": trend(last.Ingresos, prev.Ingresos),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Campaña"), use_container_width=True)
    else:
        st.info("Se necesitan al menos 2 semanas de datos para calcular tendencias.")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.markdown("---")
    section("Heatmap — Rendimiento Semanal por Campaña")
    hm_metric = st.selectbox("Métrica heatmap", ["CPL (€)", "Leads Válidos", "Entrevistas", "ROAS"], key="hm")
    fig_hm = chart_heatmap_campanas(df, hm_metric if hm_metric != "ROAS" else "ROAS")
    if fig_hm:
        st.caption("Verde = valor bajo (mejor para CPL) / Rojo = valor alto.")
        st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False}, key="heatmap_campanas")


def tab_perdidas(df: pd.DataFrame):
    """Tab 4: Análisis de pérdidas y embudo de conversión."""
    c1, c2 = st.columns([1, 1])
    with c1:
        section("Embudo de Conversión Global")
        st.plotly_chart(chart_embudo(df), use_container_width=True, config={"displayModeBar": False}, key="embudo")
    with c2:
        section("Motivos de Pérdida")
        fig_loss = chart_motivos_perdida(df)
        if fig_loss:
            st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": False}, key="motivos_perdida")
        else:
            st.info("Sin datos de motivos de pérdida para este filtro.")

    section("Pérdidas semanales vs Entrevistas vs Matrículas")
    st.plotly_chart(chart_perdida_por_semana(df), use_container_width=True, config={"displayModeBar": False}, key="perdida_semana")

    section("Análisis de pérdidas por campaña")
    _agg = {"Leads": ("Leads Válidos", "sum"), "Perdidos": ("Perdidos", "sum")}
    for alias, col in [("No_Valido", "No válido"), ("Precio", "No tiene dinero"),
                       ("Competencia", "Matriculado en otra escuela"), ("Ilocalizado", "Ilocalizado")]:
        if col in df.columns:
            _agg[alias] = (col, "sum")
    loss_sum = df.groupby("ID_Campaña").agg(**_agg).reset_index()
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

    # Análisis Google Ads
    goog = df[df["Canal"].str.contains("google", case=False, na=False)]
    if not goog.empty:
        section("Google Ads — Métricas Clave")
        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
        with gc1:
            kpi_card("🔍", "Inversión", fmt_eur(goog["Inversión (€)"].sum()), "Google Ads")
        with gc2:
            kpi_card("👥", "Leads Válidos", fmt_num(goog["Leads Válidos"].sum()), "total")
        with gc3:
            inv_g = goog["Inversión (€)"].sum()
            leads_g = goog["Leads Válidos"].sum()
            cpl_g = inv_g / leads_g if leads_g > 0 else None
            kpi_card("💶", "CPL Medio", fmt_eur(cpl_g) if cpl_g else "—", "coste por lead")
        with gc4:
            kpi_card("🎓", "Matriculados", fmt_num(goog["Matriculados"].sum()), "Google Ads")
        with gc5:
            kpi_card("💵", "Ingresos", fmt_eur(goog["Ingresos (€)"].sum()), "Google Ads")

    # Análisis Meta Ads
    meta = df[df["Canal"].str.contains("meta|fb|ig", case=False, na=False)]
    if not meta.empty:
        section("Meta Ads — Métricas Clave")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            kpi_card("📘", "Inversión", fmt_eur(meta["Inversión (€)"].sum()), "Meta Ads")
        with mc2:
            kpi_card("👥", "Leads Válidos", fmt_num(meta["Leads Válidos"].sum()), "total")
        with mc3:
            inv_m = meta["Inversión (€)"].sum()
            leads_m = meta["Leads Válidos"].sum()
            cpl_m = inv_m / leads_m if leads_m > 0 else None
            kpi_card("💶", "CPL Medio", fmt_eur(cpl_m) if cpl_m else "—", "coste por lead")
        with mc4:
            kpi_card("🎓", "Matriculados", fmt_num(meta["Matriculados"].sum()), "Meta Ads")
        with mc5:
            kpi_card("💵", "Ingresos", fmt_eur(meta["Ingresos (€)"].sum()), "Meta Ads")


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

    int_cols = ["Contactos", "Leads Válidos", "Entrevistas", "Matriculados", "Perdidos"]
    fmt_int = lambda x: f"{int(x):,}" if not pd.isna(x) else "—"
    fmt_dict = {c: fmt_int for c in int_cols if c in display.columns}
    st.dataframe(display.style.format(fmt_dict), use_container_width=True, height=500)

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

    logo_path = os.path.join(os.path.dirname(__file__), "Pontia_Logo_Isotipo_Black.png")
    sc1, sc2, sc3 = st.sidebar.columns([1, 0.8, 1])
    with sc2:
        if os.path.exists(logo_path):
            st.image(logo_path, width=50)
    st.sidebar.markdown(
        """
        <div class="sb-brand" style="border-top:none;padding-top:0">
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

    # Week filter mode (Phase 2A)
    filtro_semana_modo = st.sidebar.radio(
        "Modo filtro semana", ["Semana individual", "Rango de semanas"], key="filtro_semana_modo"
    )

    semana = "Todas"
    semanas_range = None
    if filtro_semana_modo == "Semana individual":
        semana = st.sidebar.selectbox("Semana", opts["semanas"], index=0)
    else:
        # Rango de semanas con select_slider
        semanas_num = sorted([int(s.replace("S", "")) for s in opts["semanas"] if s != "Todas"])
        if len(semanas_num) >= 2:
            rango = st.sidebar.select_slider(
                "Rango de semanas",
                options=semanas_num,
                value=(min(semanas_num), max(semanas_num)),
                key="semanas_range_slider",
            )
            semanas_range = list(range(rango[0], rango[1] + 1))
        elif semanas_num:
            st.sidebar.info(f"Solo hay una semana disponible: S{semanas_num[0]}")
            semanas_range = semanas_num

    canal = st.sidebar.selectbox("Canal", opts["canales"], index=0)
    programa = st.sidebar.selectbox("Programa", opts["programas"], index=0)

    # Campaign search (Phase 2B)
    search_campana = st.sidebar.text_input("🔍 Buscar campaña", placeholder="Nombre...", key="search_campana")

    st.sidebar.markdown("---")

    # Estadísticas rápidas
    df_f = apply_filters(df, semana, canal, programa, semanas_range=semanas_range)
    # Apply campaign search filter
    if search_campana and search_campana.strip():
        df_f = df_f[df_f["ID_Campaña"].str.contains(search_campana.strip(), case=False, na=False)]
    st.sidebar.markdown(
        f"""
        <div style="font-size:.75rem;color:#6B7280;line-height:1.8">
        📊 <b style="color:#EE7015">{len(df_f):,}</b> registros filtrados<br>
        📅 <b style="color:#EE7015">{df_f['Semana'].nunique()}</b> semanas con datos<br>
        🎯 <b style="color:#EE7015">{df_f['ID_Campaña'].nunique()}</b> campañas activas
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
        st.stop()

    # ── Load data ─────────────────────────────────────────────────────────────
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        with st.spinner("Cargando datos…"):
            df_all = load_data()
    loading_placeholder.empty()

    if df_all.empty:
        st.error("No se pudieron cargar los datos. Revisa la conexión al Google Sheet o el archivo Excel.")
        return

    # ── Sidebar + filtros ─────────────────────────────────────────────────────
    df_filtered = render_sidebar(df_all)

    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([3, 1])
    with h1:
        logo_b64 = _load_logo_b64()
        logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="height:28px;vertical-align:middle;margin-right:.5rem">' if logo_b64 else ""
        st.markdown(
            f"<h1 style='font-family:Manrope,sans-serif;font-weight:800;font-size:1.6rem;"
            f"color:#1F2937;margin:0'>{logo_img}PontIA · Marketing Intelligence</h1>"
            f"<p style='color:#6B7280;font-size:.85rem;margin:.15rem 0 .8rem'>Seguimiento semanal de campañas de captación</p>",
            unsafe_allow_html=True,
        )
    with h2:
        semanas_con_datos = sorted(df_all["Semana"].unique().tolist())
        ult = semanas_con_datos[-1] if semanas_con_datos else "—"
        st.markdown(
            f"<div style='text-align:right;font-family:IBM Plex Mono,monospace;font-size:.75rem;"
            f"color:#6B7280;padding-top:.5rem'>Última semana: <b style='color:#EE7015'>S{ult}</b><br>"
            f"Semanas totales: <b style='color:#EE7015'>{len(semanas_con_datos)}</b></div>",
            unsafe_allow_html=True,
        )

    # ── Load benchmarks ─────────────────────────────────────────────────────
    benchmarks = _load_benchmarks()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Resumen General",
        "🎯 Por Campaña",
        "📈 Evolución Histórica",
        "🚨 Análisis de Pérdidas",
        "📋 Datos",
    ])

    with t1:
        tab_resumen(df_filtered, df_all, benchmarks)
    with t2:
        tab_campanas(df_filtered, benchmarks)
    with t3:
        tab_historico(df_filtered)
    with t4:
        tab_perdidas(df_filtered)
    with t5:
        tab_datos(df_filtered)


if __name__ == "__main__":
    main()
