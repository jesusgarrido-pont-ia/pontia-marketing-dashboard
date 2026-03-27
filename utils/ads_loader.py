"""
ads_loader.py — Carga de datos de inversión desde Google Ads, Meta Ads y LinkedIn Ads.

Obtiene el gasto publicitario semanal por campaña desde cada plataforma.
Complementa los datos de HubSpot (que no incluyen inversión).

Requiere credenciales en st.secrets para cada plataforma activa.
"""

from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE ADS
# ══════════════════════════════════════════════════════════════════════════════

def _get_google_ads_config() -> dict:
    """Lee configuración de Google Ads desde secrets."""
    try:
        return dict(st.secrets.get("google_ads", {}))
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_google_ads_spend(days: int = 90) -> pd.DataFrame:
    """Carga inversión semanal por campaña desde Google Ads REST API.

    Usa la Google Ads REST API directamente (no la librería google-ads).
    Requiere: developer_token, client_id, client_secret, refresh_token, customer_id.

    Returns:
        DataFrame con columns: [ID_Campaña, Semana, Inversión, Canal]
    """
    config = _get_google_ads_config()
    if not config.get("refresh_token"):
        return pd.DataFrame()

    # Refresh access token
    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "refresh_token": config["refresh_token"],
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    token_resp.raise_for_status()
    access_token = token_resp.json()["access_token"]

    # Query via Google Ads REST API
    customer_id = config["customer_id"].replace("-", "")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    query = f"""
        SELECT
            campaign.name,
            metrics.cost_micros,
            segments.week
        FROM campaign
        WHERE segments.date BETWEEN '{start_date}' AND '{end_date}'
          AND campaign.status != 'REMOVED'
    """

    url = f"https://googleads.googleapis.com/v18/customers/{customer_id}/googleAds:searchStream"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "developer-token": config["developer_token"],
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json={"query": query}, timeout=30)
    resp.raise_for_status()

    rows = []
    for batch in resp.json():
        for result in batch.get("results", []):
            campaign_name = result.get("campaign", {}).get("name", "")
            cost_micros = int(result.get("metrics", {}).get("costMicros", 0))
            week_str = result.get("segments", {}).get("week", "")

            cost_eur = cost_micros / 1_000_000
            week_num = _week_from_date(week_str)

            if campaign_name and week_num > 0:
                rows.append({
                    "ID_Campaña": campaign_name,
                    "Semana": week_num,
                    "Inversión (€)": round(cost_eur, 2),
                    "Canal": "Google Ads",
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.groupby(["ID_Campaña", "Semana", "Canal"], as_index=False)["Inversión (€)"].sum()


# ══════════════════════════════════════════════════════════════════════════════
# META ADS (Facebook/Instagram)
# ══════════════════════════════════════════════════════════════════════════════

def _get_meta_ads_config() -> dict:
    try:
        return dict(st.secrets.get("meta_ads", {}))
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_meta_ads_spend(days: int = 90) -> pd.DataFrame:
    """Carga inversión semanal por campaña desde Meta Marketing API.

    Requiere: access_token, ad_account_id.

    Returns:
        DataFrame con columns: [ID_Campaña, Semana, Inversión, Canal]
    """
    config = _get_meta_ads_config()
    if not config.get("access_token") or not config.get("ad_account_id"):
        return pd.DataFrame()

    ad_account = config["ad_account_id"]
    token = config["access_token"]
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    url = f"https://graph.facebook.com/v21.0/{ad_account}/insights"
    params = {
        "access_token": token,
        "fields": "campaign_name,spend",
        "time_range": f'{{"since":"{start_date}","until":"{end_date}"}}',
        "time_increment": 7,  # weekly breakdown
        "level": "campaign",
        "limit": 500,
    }

    all_rows = []
    while url:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("data", []):
            campaign_name = item.get("campaign_name", "")
            spend = float(item.get("spend", 0))
            date_start = item.get("date_start", "")
            week_num = _week_from_date(date_start)

            if campaign_name and week_num > 0:
                all_rows.append({
                    "ID_Campaña": campaign_name,
                    "Semana": week_num,
                    "Inversión (€)": round(spend, 2),
                    "Canal": "Meta Ads (FB/IG)",
                })

        # Pagination
        paging = data.get("paging", {}).get("next")
        if paging:
            url = paging
            params = {}  # params are in the next URL
        else:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df.groupby(["ID_Campaña", "Semana", "Canal"], as_index=False)["Inversión (€)"].sum()


# ══════════════════════════════════════════════════════════════════════════════
# LINKEDIN ADS
# ══════════════════════════════════════════════════════════════════════════════

def _get_linkedin_ads_config() -> dict:
    try:
        return dict(st.secrets.get("linkedin_ads", {}))
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_linkedin_ads_spend(days: int = 90) -> pd.DataFrame:
    """Carga inversión semanal por campaña desde LinkedIn Marketing API.

    Requiere: access_token, ad_account_id.

    Returns:
        DataFrame con columns: [ID_Campaña, Semana, Inversión, Canal]
    """
    config = _get_linkedin_ads_config()
    if not config.get("access_token"):
        return pd.DataFrame()

    token = config["access_token"]
    ad_account = config.get("ad_account_id", "")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    url = "https://api.linkedin.com/v2/adAnalyticsV2"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    params = {
        "q": "analytics",
        "pivot": "CAMPAIGN",
        "dateRange.start.day": int(start_date.split("-")[2]),
        "dateRange.start.month": int(start_date.split("-")[1]),
        "dateRange.start.year": int(start_date.split("-")[0]),
        "dateRange.end.day": int(end_date.split("-")[2]),
        "dateRange.end.month": int(end_date.split("-")[1]),
        "dateRange.end.year": int(end_date.split("-")[0]),
        "timeGranularity": "WEEKLY",
        "fields": "costInLocalCurrency",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()

    rows = []
    for element in resp.json().get("elements", []):
        campaign_urn = element.get("pivotValue", "")
        cost = float(element.get("costInLocalCurrency", 0))
        date_range = element.get("dateRange", {})
        start = date_range.get("start", {})
        if start:
            date_str = f'{start.get("year")}-{start.get("month"):02d}-{start.get("day"):02d}'
            week_num = _week_from_date(date_str)
            if week_num > 0:
                rows.append({
                    "ID_Campaña": campaign_urn,  # Necesita mapeo a nombre
                    "Semana": week_num,
                    "Inversión (€)": round(cost, 2),
                    "Canal": "LinkedIn Ads",
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.groupby(["ID_Campaña", "Semana", "Canal"], as_index=False)["Inversión (€)"].sum()


# ══════════════════════════════════════════════════════════════════════════════
# COMBINADOR
# ══════════════════════════════════════════════════════════════════════════════

def _week_from_date(date_str: str) -> int:
    """Extrae número de semana ISO de una fecha string."""
    try:
        dt = pd.to_datetime(date_str)
        return dt.isocalendar()[1]
    except Exception:
        return 0


@st.cache_data(ttl=300, show_spinner=False)
def load_all_ads_spend(days: int = 90) -> pd.DataFrame:
    """Combina inversión de todas las plataformas de ads.

    Returns:
        DataFrame con columns: [ID_Campaña, Semana, Inversión (€), Canal]
    """
    dfs = []

    try:
        df_google = load_google_ads_spend(days)
        if not df_google.empty:
            dfs.append(df_google)
    except Exception:
        pass

    try:
        df_meta = load_meta_ads_spend(days)
        if not df_meta.empty:
            dfs.append(df_meta)
    except Exception:
        pass

    try:
        df_linkedin = load_linkedin_ads_spend(days)
        if not df_linkedin.empty:
            dfs.append(df_linkedin)
    except Exception:
        pass

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
