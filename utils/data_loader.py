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
from urllib.parse import quote

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Seguimiento_Campanas_Semanal.xlsx")
SHEET_DATA = "02_DATA_LOG"
SHEET_CAMPAIGNS = "01_CAMPAÑAS"

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

def _get_hubspot_token() -> str:
    """Verifica si hay un token de HubSpot configurado."""
    try:
        return st.secrets.get("hubspot", {}).get("access_token", "").strip()
    except Exception:
        return ""


@st.cache_data(ttl=300, show_spinner="Cargando datos...")
def load_data() -> pd.DataFrame:
    """Devuelve el DataFrame de campañas limpio y enriquecido.

    Si HubSpot está configurado → usa HubSpot + Ads APIs.
    Si no → usa Google Sheets.
    Los datos se cachean 5 minutos, así que las APIs solo se llaman 1 vez cada 5 min.
    """
    hubspot_token = _get_hubspot_token()
    if hubspot_token:
        try:
            from utils.hubspot_loader import load_hubspot_data
            from utils.ads_loader import load_all_ads_spend

            df_hs = load_hubspot_data()
            if not df_hs.empty:
                # Merge con datos de inversión de Ads APIs
                try:
                    df_ads = load_all_ads_spend()
                    if not df_ads.empty:
                        df_hs = _merge_investment_data(df_hs, df_ads)
                except Exception:
                    pass  # Ads APIs opcionales
                return _process(df_hs)
            # HubSpot devolvió datos vacíos — caer a Google Sheets
            st.info("📡 HubSpot no devolvió datos (¿no hay contactos con utm_campaign?). Usando Google Sheets.")
        except Exception as e:
            st.error(f"❌ Error HubSpot: {e}")
            st.info("Usando Google Sheets como respaldo...")

    # Google Sheets
    spreadsheet_id = _get_spreadsheet_id()
    if not spreadsheet_id:
        raise RuntimeError("No hay fuente de datos configurada.")
    return _from_sheets(spreadsheet_id)


def _merge_investment_data(df_hubspot: pd.DataFrame, df_ads: pd.DataFrame) -> pd.DataFrame:
    """Merge inversión de Ads APIs con datos de leads de HubSpot."""
    if df_ads.empty:
        return df_hubspot

    # Agrupar inversión por campaña y semana
    ads_grouped = df_ads.groupby(["ID_Campaña", "Semana"], as_index=False)["Inversión (€)"].sum()

    # Merge
    df = df_hubspot.merge(
        ads_grouped[["ID_Campaña", "Semana", "Inversión (€)"]],
        on=["ID_Campaña", "Semana"],
        how="left",
        suffixes=("_hs", "_ads"),
    )

    # Usar inversión de Ads si existe, sino la de HubSpot (que es 0)
    if "Inversión (€)_ads" in df.columns:
        df["Inversión (€)"] = df["Inversión (€)_ads"].fillna(df.get("Inversión (€)_hs", 0))
        df = df.drop(columns=[c for c in df.columns if c.endswith("_ads") or c.endswith("_hs")])

    # Recalcular CPL con inversión real
    df["CPL (€)"] = np.where(
        df["Leads Válidos"].fillna(0) > 0,
        df["Inversión (€)"].fillna(0) / df["Leads Válidos"],
        0,
    )
    # Recalcular Coste Entrevista
    df["Coste Entrevista (€)"] = np.where(
        df["Entrevistas"].fillna(0) > 0,
        df["Inversión (€)"].fillna(0) / df["Entrevistas"],
        0,
    )

    return df


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


# ── Carga hoja de campañas (estado) ───────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_campaign_status() -> pd.DataFrame:
    """Carga la hoja 01.CAMPAÑAS para obtener el estado de cada campaña."""
    spreadsheet_id = _get_spreadsheet_id()
    if not spreadsheet_id:
        return pd.DataFrame()
    try:
        sheet_encoded = quote(SHEET_CAMPAIGNS)
        url = (
            f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            f"/gviz/tq?tqx=out:csv&sheet={sheet_encoded}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df_camp = pd.read_csv(StringIO(resp.text), header=0)
        df_camp.columns = [c.strip() for c in df_camp.columns]
        # Buscar columna de estado (puede llamarse Estado, Status, etc.)
        estado_col = None
        for col in df_camp.columns:
            if col.lower() in ("estado", "status", "estado campaña"):
                estado_col = col
                break
        # Buscar columna de ID de campaña
        id_col = None
        for col in df_camp.columns:
            if col.lower() in ("id_campaña", "id campaña", "campaña", "campaign", "nombre", "id"):
                id_col = col
                break
        if estado_col and id_col:
            result = df_camp[[id_col, estado_col]].copy()
            result.columns = ["ID_Campaña", "Estado_Campaña"]
            result["ID_Campaña"] = result["ID_Campaña"].astype(str).str.strip()
            result["Estado_Campaña"] = result["Estado_Campaña"].astype(str).str.strip()
            # Filtrar filas vacías
            result = result[result["ID_Campaña"] != ""]
            result = result[result["ID_Campaña"] != "nan"]
            return result
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── Opciones de filtro ─────────────────────────────────────────────────────

def get_filter_options(df: pd.DataFrame) -> dict:
    semanas = sorted(df["Semana"].unique().tolist())
    semana_labels = ["Todas"] + [f"S{s}" for s in semanas if s > 0]
    campanas = ["Todas"] + sorted(df["ID_Campaña"].unique().tolist())
    canales = ["Todos"] + sorted(df["Canal"].dropna().unique().tolist())
    programas = ["Todos"] + sorted(df["Programa"].dropna().unique().tolist())
    # Estados de campaña
    estados = ["Todos"]
    if "Estado_Campaña" in df.columns:
        estados += sorted(df["Estado_Campaña"].dropna().unique().tolist())
    return {
        "semanas": semana_labels,
        "campanas": campanas,
        "canales": canales,
        "programas": programas,
        "estados": estados,
    }


def apply_filters(df: pd.DataFrame, semana: str, canal=None, programa=None, semanas_range=None, estado=None) -> pd.DataFrame:
    out = df.copy()
    if semanas_range is not None:
        out = out[out["Semana"].isin([float(s) for s in semanas_range])]
    elif semana and semana != "Todas":
        num = int(semana.replace("S", ""))
        out = out[out["Semana"] == num]
    # Canal: accept string (legacy) or list
    if isinstance(canal, list) and canal:
        out = out[out["Canal"].isin(canal)]
    elif isinstance(canal, str) and canal and canal != "Todos":
        out = out[out["Canal"] == canal]
    # Programa: accept string (legacy) or list
    if isinstance(programa, list) and programa:
        out = out[out["Programa"].isin(programa)]
    elif isinstance(programa, str) and programa and programa != "Todos":
        out = out[out["Programa"] == programa]
    # Estado de campaña
    if isinstance(estado, list) and estado:
        out = out[out["Estado_Campaña"].isin(estado)]
    elif isinstance(estado, str) and estado and estado != "Todos":
        out = out[out["Estado_Campaña"] == estado]
    return out
