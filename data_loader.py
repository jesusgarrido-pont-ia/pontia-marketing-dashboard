"""
data_loader.py — Carga y preprocesado de datos de campañas Pontia.

Fuentes (por prioridad):
  1. Google Sheets (CSV export) — cuando SPREADSHEET_ID está configurado
  2. Archivo Excel local           — fallback para desarrollo
"""

import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Seguimiento_Campanas_Semanal.xlsx")
SHEET_DATA = "02_DATA_LOG"

# Mapeo de nombres de columna del Google Sheet → nombres esperados por el dashboard
# (el Google Sheet no tiene acentos/€ en algunos nombres)
_COLUMN_REMAP = {
    "Fecha de análisis":      "Fecha de Análisis",
    "Fecha de Analisis":      "Fecha de Análisis",
    "Inversion":              "Inversión (€)",
    "Inversión":              "Inversión (€)",
    "Leads Validos":          "Leads Válidos",
    "CPL":                    "CPL (€)",
    "Coste Entrevista":       "Coste Entrevista (€)",
    "Ingresos":               "Ingresos (€)",
    "No interesa(Otros)":     "No interesa (Otros)",
    "No interesa (Otros)":    "No interesa (Otros)",  # ya correcto
    "Matriculado en otra":    "Matriculado en otra escuela",
}


# ── Carga principal ────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_data() -> pd.DataFrame:
    """Devuelve el DataFrame de campañas limpio y enriquecido. Fuente: Google Sheets."""
    spreadsheet_id = _get_spreadsheet_id()
    if not spreadsheet_id:
        raise RuntimeError("Google Sheets no configurado. Añade spreadsheet_id en secrets.")
    return _from_sheets(spreadsheet_id)


def _get_spreadsheet_id() -> str:
    try:
        sid = st.secrets.get("google_sheets", {}).get("spreadsheet_id", "")
        return sid.strip() if sid else ""
    except Exception:
        return ""


def _from_sheets(spreadsheet_id: str) -> pd.DataFrame:
    try:
        gs = st.secrets.get("google_sheets", {})
        gid = gs.get("sheet_gid", "")
        sheet_name = gs.get("sheet_name", "02_DATA_LOG")
    except Exception:
        gid = ""
        sheet_name = "02_DATA_LOG"

    if gid:
        url = (
            f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            f"/gviz/tq?tqx=out:csv&gid={gid}"
        )
    else:
        url = (
            f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        )

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), header=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=_COLUMN_REMAP)
    df = _clean_euro_columns(df)

    return _process(df)


def _clean_euro_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia columnas con formato monetario europeo de Google Sheets.
    Maneja tanto '120,33 €' → 120.33 como '1.500,00 €' → 1500.0
    """
    for col in df.select_dtypes(include="object").columns:
        s = df[col].astype(str).str.strip().str.replace(r"[€%]", "", regex=True).str.strip()
        # Formato europeo: el punto es separador de miles, la coma es decimal
        # Ejemplo: "1.500,00" → quitar punto de miles → "1500,00" → "1500.00"
        has_comma = s.str.contains(",", na=False)
        s = s.where(~has_comma, s.str.replace(".", "", regex=False))
        s = s.str.replace(",", ".", regex=False)
        numeric = pd.to_numeric(s, errors="coerce")
        if df[col].notna().sum() > 0:
            ratio = numeric.notna().sum() / df[col].notna().sum()
            if ratio >= 0.4:
                df[col] = numeric
    return df


def _from_excel() -> pd.DataFrame:
    path = os.path.abspath(EXCEL_PATH)
    df = pd.read_excel(path, sheet_name=SHEET_DATA, header=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=_COLUMN_REMAP)
    return _process(df)


# ── Procesado ─────────────────────────────────────────────────────────────

def _process(df: pd.DataFrame) -> pd.DataFrame:
    # Eliminar filas sin campaña ni fecha
    df = df.dropna(subset=["ID_Campaña", "Fecha de Análisis"]).copy()
    df = df[df["ID_Campaña"].astype(str).str.strip() != ""]

    # Fechas y semana
    df["Fecha de Análisis"] = pd.to_datetime(df["Fecha de Análisis"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Fecha de Análisis"])

    df["Semana"] = pd.to_numeric(df["Semana"], errors="coerce").fillna(0).astype(int)
    df["Semana_label"] = "S" + df["Semana"].astype(str)

    # Columnas numéricas
    NUM_COLS = [
        "Inversión (€)", "Contactos", "Leads Válidos", "CPL (€)",
        "Coste Entrevista (€)", "Entrevistas", "Matriculados", "Ingresos (€)",
        "Perdidos", "Exploración", "Consideración", "Decisión",
        "Leads Pre-Cualificados", "Leads Post-Cualificados",
        "No es lo que buscaba", "No válido", "No tiene dinero",
        "No interesa (Otros)", "Matriculado en otra escuela",
        "Próxima convocatoria", "Anciano", "Busca Certificación",
        "Busca otra metodología", "Criterios de Admisión",
        "Ilocalizado", "No tiene tiempo", "Desistimiento",
        "% Lead→Entrevista", "% Entrevista→Matrícula",
        "% Post-Cualificados", "% Alta Intención", "% Pérdida",
        "Ingreso/Lead (€)", "% Pérdida Precio", "% Pérdida Producto",
        "% Pérdida Competencia", "% Ilocalizados", "CPA (€)",
    ]
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ROAS calculado
    df["ROAS"] = np.where(
        df["Inversión (€)"].fillna(0) > 0,
        df["Ingresos (€)"].fillna(0) / df["Inversión (€)"],
        np.nan,
    )

    # % Alta Intención recalculado cuando falta
    if "% Alta Intención" not in df.columns:
        df["% Alta Intención"] = np.nan
    mask = df["% Alta Intención"].isna() & (df["Leads Válidos"].fillna(0) > 0)
    if mask.any() and "Consideración" in df.columns and "Decisión" in df.columns:
        df.loc[mask, "% Alta Intención"] = (
            (df.loc[mask, "Consideración"].fillna(0) + df.loc[mask, "Decisión"].fillna(0))
            / df.loc[mask, "Leads Válidos"]
        )

    # Conv. Lead→Matrícula
    df["Conv. Lead→Mat."] = np.where(
        df["Leads Válidos"].fillna(0) > 0,
        df["Matriculados"].fillna(0) / df["Leads Válidos"],
        np.nan,
    )

    # Tipo de canal simplificado
    df["Tipo Canal"] = df["Canal"].apply(_tipo_canal)

    return df.reset_index(drop=True)


def _tipo_canal(canal: str) -> str:
    c = str(canal).lower()
    if "meta" in c or "fb" in c or "ig" in c:
        return "Meta Ads"
    if "google" in c:
        return "Google Ads"
    if "orgánico" in c or "organico" in c or "seo" in c:
        return "Orgánico / SEO"
    if "youtube" in c:
        return "YouTube Ads"
    if "linkedin" in c:
        return "LinkedIn Ads"
    return str(canal)


# ── Opciones de filtro ─────────────────────────────────────────────────────

def get_filter_options(df: pd.DataFrame) -> dict:
    semanas = sorted(df["Semana"].unique().tolist())
    semana_labels = ["Todas"] + [f"S{s}" for s in semanas if s > 0]
    campanas = ["Todas"] + sorted(df["ID_Campaña"].unique().tolist())
    canales = ["Todos"] + sorted(df["Canal"].dropna().unique().tolist())
    programas = ["Todos"] + sorted(df["Programa"].dropna().unique().tolist())
    return {
        "semanas": semana_labels,
        "campanas": campanas,
        "canales": canales,
        "programas": programas,
    }


def apply_filters(df: pd.DataFrame, semana: str, canal: str, programa: str) -> pd.DataFrame:
    out = df.copy()
    if semana and semana != "Todas":
        num = int(semana.replace("S", ""))
        out = out[out["Semana"] == num]
    if canal and canal != "Todos":
        out = out[out["Canal"] == canal]
    if programa and programa != "Todos":
        out = out[out["Programa"] == programa]
    return out
