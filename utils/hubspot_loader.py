"""
hubspot_loader.py — Carga de datos de HubSpot CRM via API REST v3.

Obtiene contactos, deals, etapas del pipeline y motivos de pérdida.
Transforma los datos al mismo formato que el Google Sheet (02_DATA_LOG).

Requiere: Private App Access Token en st.secrets["hubspot"]["access_token"]
Plan: HubSpot Starter (100 req/10seg, 250k daily)
"""

import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

HUBSPOT_BASE = "https://api.hubapi.com"

# Mapeo de utm_source a tipo de canal (configurable)
_DEFAULT_UTM_SOURCE_MAP = {
    "facebook": "Meta Ads (FB/IG)",
    "fb": "Meta Ads (FB/IG)",
    "instagram": "Meta Ads (FB/IG)",
    "ig": "Meta Ads (FB/IG)",
    "meta": "Meta Ads (FB/IG)",
    "google": "Google Ads",
    "goog": "Google Ads",
    "youtube": "YouTube Ads",
    "linkedin": "LinkedIn Ads",
    "organic": "Orgánico / SEO",
    "direct": "Orgánico / SEO",
}


def _get_token() -> str:
    """Obtiene el token de acceso de HubSpot desde Streamlit secrets."""
    try:
        return st.secrets.get("hubspot", {}).get("access_token", "").strip()
    except Exception:
        return ""


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _rate_limit_pause():
    """Pausa mínima entre requests para respetar rate limits (100/10seg)."""
    time.sleep(0.12)


# ── Fetch de contactos ────────────────────────────────────────────────────────

def _fetch_all_contacts(token: str, since_days: int = 180) -> list[dict]:
    """Pagina contactos con utm_campaign relleno (solo leads de campañas)."""
    props = [
        "email", "createdate", "lifecyclestage", "hs_lead_status",
        "utm_campaign", "utm_source", "utm_medium",
    ]
    # Usar Search API (POST) — filtrar solo contactos CON utm_campaign
    # Esto reduce 30.000 contactos a solo los que vienen de campañas (~1.000-2.000)
    url = f"{HUBSPOT_BASE}/crm/v3/objects/contacts/search"
    since = datetime.utcnow() - timedelta(days=since_days)
    since_ms = str(int(since.timestamp() * 1000))

    all_contacts = []
    after = 0

    while True:
        body = {
            "limit": 200,
            "properties": props,
            "filterGroups": [{
                "filters": [
                    {
                        "propertyName": "createdate",
                        "operator": "GTE",
                        "value": since_ms,
                    },
                    {
                        "propertyName": "utm_campaign",
                        "operator": "HAS_PROPERTY",
                    },
                ]
            }],
            "after": after,
        }

        resp = requests.post(url, headers=_headers(token), json=body, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(
                f"HubSpot API error (status {resp.status_code}). Verifica el access_token en Secrets."
            )
        data = resp.json()

        for item in data.get("results", []):
            p = item.get("properties", {})
            all_contacts.append({
                "id": item["id"],
                "email": p.get("email", ""),
                "createdate": p.get("createdate", ""),
                "lifecyclestage": p.get("lifecyclestage", ""),
                "hs_lead_status": p.get("hs_lead_status", ""),
                "utm_campaign": p.get("utm_campaign", ""),
                "utm_source": p.get("utm_source", ""),
                "utm_medium": p.get("utm_medium", ""),
            })

        paging = data.get("paging", {}).get("next", {})
        next_after = paging.get("after")
        if not next_after:
            break
        after = int(next_after)
        _rate_limit_pause()

    return all_contacts


# ── Fetch de deals ────────────────────────────────────────────────────────────

def _fetch_all_deals(token: str, since_days: int = 180) -> list[dict]:
    """Pagina todos los deals creados en los últimos since_days días."""
    props = [
        "dealname", "amount", "dealstage", "closedate", "createdate",
        "pipeline", "closed_lost_reason",
    ]
    url = f"{HUBSPOT_BASE}/crm/v3/objects/deals"
    since = datetime.utcnow() - timedelta(days=since_days)
    since_ms = int(since.timestamp() * 1000)

    all_deals = []
    after = None

    while True:
        params = {
            "limit": 100,
            "properties": ",".join(props),
        }
        if after:
            params["after"] = after

        resp = requests.get(url, headers=_headers(token), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("results", []):
            p = item.get("properties", {})
            all_deals.append({
                "id": item["id"],
                "dealname": p.get("dealname", ""),
                "amount": p.get("amount", ""),
                "dealstage": p.get("dealstage", ""),
                "closedate": p.get("closedate", ""),
                "createdate": p.get("createdate", ""),
                "pipeline": p.get("pipeline", ""),
                "closed_lost_reason": p.get("closed_lost_reason", ""),
            })

        paging = data.get("paging", {}).get("next", {})
        after = paging.get("after")
        if not after:
            break
        _rate_limit_pause()

    return all_deals


# ── Asociaciones deal → contact ───────────────────────────────────────────────

def _fetch_deal_contact_associations(token: str, deal_ids: list[str]) -> dict:
    """Mapea deal_id → contact_id usando batch association API."""
    url = f"{HUBSPOT_BASE}/crm/v3/associations/deals/contacts/batch/read"
    mapping = {}

    # Procesar en lotes de 100
    for i in range(0, len(deal_ids), 100):
        batch = deal_ids[i:i + 100]
        body = {"inputs": [{"id": did} for did in batch]}

        resp = requests.post(url, headers=_headers(token), json=body, timeout=30)
        if resp.status_code == 200:
            for result in resp.json().get("results", []):
                deal_id = result.get("from", {}).get("id")
                contacts = result.get("to", [])
                if deal_id and contacts:
                    mapping[deal_id] = contacts[0].get("id")
        _rate_limit_pause()

    return mapping


# ── Pipeline stages ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_pipeline_stages(_token: str) -> dict:
    """Obtiene mapeo stage_id → stage_label del pipeline de deals."""
    url = f"{HUBSPOT_BASE}/crm/v3/pipelines/deals"
    resp = requests.get(url, headers=_headers(_token), timeout=15)
    resp.raise_for_status()

    stages = {}
    for pipeline in resp.json().get("results", []):
        for stage in pipeline.get("stages", []):
            stages[stage["id"]] = stage["label"]
    return stages


# ── Transformación a formato semanal ──────────────────────────────────────────

def _map_utm_to_canal(utm_source: str, utm_medium: str) -> str:
    """Mapea utm_source/medium al tipo de canal del dashboard."""
    src = str(utm_source).lower().strip()
    med = str(utm_medium).lower().strip()

    for key, canal in _DEFAULT_UTM_SOURCE_MAP.items():
        if key in src or key in med:
            return canal

    if "cpc" in med or "paid" in med:
        return "Google Ads"
    return "Orgánico / SEO"


def _week_number(date_str: str) -> int:
    """Extrae número de semana ISO de un string de fecha."""
    try:
        dt = pd.to_datetime(date_str)
        return dt.isocalendar()[1]
    except Exception:
        return 0


def _transform_to_weekly(
    contacts: list[dict],
    deals: list[dict],
    deal_contact_map: dict,
    stages: dict,
    stage_mapping: dict | None = None,
) -> pd.DataFrame:
    """Transforma datos crudos de HubSpot al formato semanal del Google Sheet.

    Args:
        contacts: Lista de contactos con propiedades.
        deals: Lista de deals con propiedades.
        deal_contact_map: Mapeo deal_id → contact_id.
        stages: Mapeo stage_id → stage_label.
        stage_mapping: Mapeo custom de stage labels a etapas del funnel.
    """
    if not contacts:
        return pd.DataFrame()

    # Mapeo por defecto de etapas
    if stage_mapping is None:
        stage_mapping = {
            "exploración": "Exploración",
            "consideración": "Consideración",
            "decisión": "Decisión",
        }

    # ── Contactos → DataFrame ─────────────────────────────────────────────
    df_contacts = pd.DataFrame(contacts)
    df_contacts["week"] = df_contacts["createdate"].apply(_week_number)
    df_contacts["canal"] = df_contacts.apply(
        lambda r: _map_utm_to_canal(r["utm_source"], r["utm_medium"]), axis=1
    )
    df_contacts["campaign"] = df_contacts["utm_campaign"].fillna("sin_campaña").str.strip()
    df_contacts = df_contacts[df_contacts["campaign"] != ""]
    df_contacts = df_contacts[df_contacts["campaign"] != "sin_campaña"]

    # ── Deals → enriquecer con contact info ───────────────────────────────
    contact_lookup = {c["id"]: c for c in contacts}
    deal_records = []
    for d in deals:
        contact_id = deal_contact_map.get(d["id"])
        contact = contact_lookup.get(contact_id, {}) if contact_id else {}
        stage_label = stages.get(d["dealstage"], d["dealstage"]).lower()

        deal_records.append({
            "deal_id": d["id"],
            "campaign": contact.get("utm_campaign", ""),
            "week": _week_number(d.get("createdate", "")),
            "stage_label": stage_label,
            "amount": float(d["amount"]) if d.get("amount") else 0,
            "closed_lost_reason": d.get("closed_lost_reason", ""),
            "is_won": "closedwon" in stage_label.replace(" ", "").replace("-", ""),
            "is_lost": "closedlost" in stage_label.replace(" ", "").replace("-", ""),
        })

    df_deals = pd.DataFrame(deal_records) if deal_records else pd.DataFrame()

    # ── Agregar por (campaña, semana) ─────────────────────────────────────
    groups = df_contacts.groupby(["campaign", "week", "canal"])
    rows = []

    for (camp, week, canal), gc in groups:
        if week == 0:
            continue

        n_contacts = len(gc)
        n_leads_valid = len(gc[gc["hs_lead_status"] != "unqualified"])

        # Deals de esta campaña y semana
        camp_deals = df_deals[
            (df_deals["campaign"] == camp) & (df_deals["week"] == week)
        ] if not df_deals.empty else pd.DataFrame()

        # Etapas del funnel
        n_exploracion = 0
        n_consideracion = 0
        n_decision = 0
        n_entrevistas = 0
        n_matriculados = 0
        n_perdidos = 0
        ingresos = 0
        loss_reasons = {}

        if not camp_deals.empty:
            for _, deal in camp_deals.iterrows():
                sl = str(deal["stage_label"])
                for key, funnel_stage in stage_mapping.items():
                    if key in sl:
                        if funnel_stage == "Exploración":
                            n_exploracion += 1
                        elif funnel_stage == "Consideración":
                            n_consideracion += 1
                            n_entrevistas += 1
                        elif funnel_stage == "Decisión":
                            n_decision += 1
                            n_entrevistas += 1

                if deal["is_won"]:
                    n_matriculados += 1
                    ingresos += deal["amount"]
                elif deal["is_lost"]:
                    n_perdidos += 1
                    reason = str(deal["closed_lost_reason"]).strip()
                    if reason:
                        loss_reasons[reason] = loss_reasons.get(reason, 0) + 1

        rows.append({
            "ID_Campaña": camp,
            "Semana": int(week),
            "Canal": canal,
            "Programa": "",  # Se inferirá del naming o property custom
            "Contactos": n_contacts,
            "Leads Válidos": n_leads_valid,
            "Inversión (€)": 0,  # Viene de Ads APIs
            "CPL (€)": 0,  # Se calculará cuando haya inversión
            "Exploración": n_exploracion,
            "Consideración": n_consideracion,
            "Decisión": n_decision,
            "Entrevistas": n_entrevistas,
            "Matriculados": n_matriculados,
            "Ingresos (€)": ingresos,
            "Perdidos": n_perdidos,
            "Coste Entrevista (€)": 0,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Función principal ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_hubspot_data(since_days: int = 180) -> pd.DataFrame:
    """Carga y transforma datos de HubSpot al formato del Google Sheet.

    Returns:
        DataFrame con el mismo schema que 02_DATA_LOG (sin inversión).
    """
    token = _get_token()
    if not token:
        raise RuntimeError("HubSpot access token no configurado en secrets.")

    contacts = _fetch_all_contacts(token, since_days)
    deals = _fetch_all_deals(token, since_days)

    deal_ids = [d["id"] for d in deals]
    deal_contact_map = _fetch_deal_contact_associations(token, deal_ids) if deal_ids else {}

    stages = _fetch_pipeline_stages(token)

    return _transform_to_weekly(contacts, deals, deal_contact_map, stages)


# ── Diagnóstico completo de HubSpot ───────────────────────────────────────────

def diagnose_hubspot() -> dict:
    """Diagnóstico: lista todas las propiedades de contactos y deals,
    las etapas del pipeline, y una muestra de contactos con sus valores.

    Devuelve un dict con toda la info necesaria para mapear HubSpot → Google Sheet.
    """
    token = _get_token()
    if not token:
        return {"error": "No HubSpot token configured"}

    result = {}

    # 1. Propiedades de contactos
    try:
        resp = requests.get(
            f"{HUBSPOT_BASE}/crm/v3/properties/contacts",
            headers=_headers(token), timeout=15,
        )
        resp.raise_for_status()
        props = resp.json().get("results", [])
        result["contact_properties"] = [
            {
                "name": p["name"],
                "label": p["label"],
                "type": p["type"],
                "group": p.get("groupName", ""),
            }
            for p in props
        ]
    except Exception as e:
        result["contact_properties_error"] = str(e)

    # 2. Propiedades de deals
    try:
        resp = requests.get(
            f"{HUBSPOT_BASE}/crm/v3/properties/deals",
            headers=_headers(token), timeout=15,
        )
        resp.raise_for_status()
        props = resp.json().get("results", [])
        result["deal_properties"] = [
            {
                "name": p["name"],
                "label": p["label"],
                "type": p["type"],
                "group": p.get("groupName", ""),
            }
            for p in props
        ]
    except Exception as e:
        result["deal_properties_error"] = str(e)

    # 3. Pipeline y etapas
    try:
        stages = _fetch_pipeline_stages(token)
        result["pipeline_stages"] = stages
    except Exception as e:
        result["pipeline_stages_error"] = str(e)

    # 4. Muestra de contactos RECIENTES con TODAS las propiedades
    # Para descubrir dónde HubSpot guarda el nombre de campaña de ads
    try:
        # Obtener nombres de TODAS las propiedades
        all_prop_names = [p["name"] for p in result.get("contact_properties", [])]

        # Traer contactos recientes (últimos 30 días) de pago con TODAS las propiedades
        since_30d = str(int((datetime.utcnow() - timedelta(days=30)).timestamp() * 1000))

        for source_type in ["PAID_SEARCH", "PAID_SOCIAL"]:
            body = {
                "limit": 10,
                "properties": all_prop_names[:200],  # API limit ~200 props per request
                "sorts": [{"propertyName": "createdate", "direction": "DESCENDING"}],
                "filterGroups": [{
                    "filters": [
                        {
                            "propertyName": "hs_analytics_source",
                            "operator": "EQ",
                            "value": source_type,
                        },
                        {
                            "propertyName": "createdate",
                            "operator": "GTE",
                            "value": since_30d,
                        },
                    ]
                }],
            }
            resp = requests.post(
                f"{HUBSPOT_BASE}/crm/v3/objects/contacts/search",
                headers=_headers(token), json=body, timeout=15,
            )
            if resp.status_code == 200:
                contacts_raw = resp.json().get("results", [])
                # Solo guardar propiedades que tienen valor (no vacías)
                contacts = []
                for item in contacts_raw:
                    props = item.get("properties", {})
                    filled = {k: v for k, v in props.items() if v and v.strip()}
                    contacts.append(filled)
                result[f"sample_contacts_{source_type}"] = contacts

        # También traer las actividades de ads (ad interactions) del primer contacto
        for source_type in ["PAID_SEARCH", "PAID_SOCIAL"]:
            key = f"sample_contacts_{source_type}"
            if key in result and result[key]:
                contact_id = result[key][0].get("hs_object_id")
                if contact_id:
                    # Obtener timeline/actividad de este contacto
                    resp = requests.get(
                        f"{HUBSPOT_BASE}/crm/v3/objects/contacts/{contact_id}/associations/deals",
                        headers=_headers(token), timeout=15,
                    )
                    if resp.status_code == 200:
                        result[f"contact_{source_type}_deals"] = resp.json().get("results", [])

    except Exception as e:
        result["sample_contacts_error"] = str(e)

    # 5. Muestra de deals con propiedades clave
    _DEAL_KEY_PROPS = [
        "dealname", "dealstage", "amount", "pipeline", "closedate", "createdate",
        "closed_lost_reason", "closed_won_reason", "deal_master", "master_matriculado",
        "hs_analytics_source", "hs_analytics_source_data_1", "hs_analytics_source_data_2",
        "hs_is_closed_won", "hs_is_closed_lost",
    ]
    try:
        resp = requests.get(
            f"{HUBSPOT_BASE}/crm/v3/objects/deals",
            headers=_headers(token),
            params={"limit": 20, "properties": ",".join(_DEAL_KEY_PROPS)},
            timeout=15,
        )
        resp.raise_for_status()
        result["sample_deals"] = [
            item.get("properties", {})
            for item in resp.json().get("results", [])
        ]
    except Exception as e:
        result["sample_deals_error"] = str(e)

    return result


def diagnose_utms(gs_campaign_ids: list[str] | None = None) -> dict:
    """Diagnóstico: compara utm_campaign de HubSpot con IDs del Google Sheet.

    Args:
        gs_campaign_ids: Lista de ID_Campaña del Google Sheet para comparar.

    Returns:
        Dict con: hubspot_utms, sheet_ids, matched, unmatched_hubspot, unmatched_sheet
    """
    token = _get_token()
    if not token:
        return {"error": "No HubSpot token configured"}

    contacts = _fetch_all_contacts(token, since_days=90)
    hs_utms = set()
    for c in contacts:
        utm = str(c.get("utm_campaign", "")).strip()
        if utm and utm != "nan":
            hs_utms.add(utm)

    result = {
        "hubspot_utms": sorted(hs_utms),
        "total_contacts": len(contacts),
        "contacts_with_utm": len([c for c in contacts if c.get("utm_campaign")]),
    }

    if gs_campaign_ids:
        gs_ids = set(gs_campaign_ids)
        matched = hs_utms & gs_ids
        result["sheet_ids"] = sorted(gs_ids)
        result["matched"] = sorted(matched)
        result["unmatched_hubspot"] = sorted(hs_utms - gs_ids)
        result["unmatched_sheet"] = sorted(gs_ids - hs_utms)
        result["match_rate"] = len(matched) / len(hs_utms) * 100 if hs_utms else 0

    return result
