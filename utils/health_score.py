"""
health_score.py — Algoritmo de Health Score para campañas de marketing.

Calcula un score 0-100 por campaña basado en:
  - Tendencia CPL (30%)
  - % Alta Intención (25%)
  - Coste por Entrevista (25%)
  - Volumen de leads (20%)
"""

import numpy as np
import pandas as pd

# ── Pesos del Health Score ────────────────────────────────────────────────────
W_CPL_TREND = 0.30
W_ALTA_INT = 0.25
W_COSTE_ENT = 0.25
W_VOLUME = 0.20

# ── Umbrales por defecto (se sobreescriben con benchmarks de config.yaml) ─────
_DEFAULTS = {
    "coste_entrevista": {"good": 60, "bad": 100},
    "cpl": {"good": 15, "review": 25, "pause": 40},
}

# ── Colores por acción ────────────────────────────────────────────────────────
ACTION_STYLES = {
    "ESCALAR": {
        "icon": "🟢", "color": "#16A34A", "bg": "rgba(22,163,74,.06)",
        "border": "rgba(22,163,74,.3)", "text": "#166534",
    },
    "MANTENER": {
        "icon": "🟡", "color": "#F59E0B", "bg": "rgba(245,158,11,.06)",
        "border": "rgba(245,158,11,.3)", "text": "#92400E",
    },
    "OPTIMIZAR": {
        "icon": "🟠", "color": "#EE7015", "bg": "rgba(238,112,21,.06)",
        "border": "rgba(238,112,21,.3)", "text": "#9A3412",
    },
    "PAUSAR": {
        "icon": "🔴", "color": "#DC2626", "bg": "rgba(239,68,68,.06)",
        "border": "rgba(239,68,68,.3)", "text": "#991B1B",
    },
    "NUEVA": {
        "icon": "🆕", "color": "#5683D2", "bg": "rgba(86,131,210,.06)",
        "border": "rgba(86,131,210,.3)", "text": "#1E3A8A",
    },
}


def compute_health_score(
    df_all: pd.DataFrame,
    current_week: int,
    benchmarks: dict | None = None,
    n_weeks: int = 4,
) -> pd.DataFrame:
    """Calcula el Health Score por campaña usando datos de las últimas n_weeks semanas.

    Args:
        df_all: DataFrame completo (sin filtrar por semana).
        current_week: Número de la semana actual (ej: 12).
        benchmarks: Dict con umbrales configurables.
        n_weeks: Número de semanas para calcular tendencias (default 4).

    Returns:
        DataFrame con una fila por campaña y columnas de score + acción.
    """
    if benchmarks is None:
        benchmarks = {}
    b_ce = benchmarks.get("coste_entrevista", _DEFAULTS["coste_entrevista"])
    b_cpl = benchmarks.get("cpl", _DEFAULTS["cpl"])

    # Excluir orgánico/SEO
    df = df_all[~df_all["Canal"].str.contains("orgánico|seo|organic", case=False, na=False)].copy()
    if df.empty:
        return pd.DataFrame()

    # Semanas disponibles ordenadas
    all_weeks = sorted(df["Semana"].unique())
    if current_week not in all_weeks:
        current_week = all_weeks[-1] if all_weeks else 0

    # Ventana de semanas para tendencia
    idx = all_weeks.index(current_week) if current_week in all_weeks else len(all_weeks) - 1
    start_idx = max(0, idx - n_weeks + 1)
    trend_weeks = all_weeks[start_idx : idx + 1]

    df_window = df[df["Semana"].isin(trend_weeks)]

    # Campañas únicas
    campaigns = df_window["ID_Campaña"].unique()
    rows = []

    for camp_id in campaigns:
        dc = df_window[df_window["ID_Campaña"] == camp_id]
        dc_current = dc[dc["Semana"] == current_week]

        programa = dc["Programa"].mode().iloc[0] if not dc["Programa"].mode().empty else "—"
        canal = dc["Canal"].mode().iloc[0] if not dc["Canal"].mode().empty else "—"

        # ── Datos por semana ──────────────────────────────────────────────
        weekly = (
            dc.groupby("Semana")
            .agg({
                "CPL (€)": "mean",
                "Leads Válidos": "sum",
                "Entrevistas": "sum",
                "Inversión (€)": "sum",
                "Consideración": "sum",
                "Decisión": "sum",
            })
            .reindex(trend_weeks)
        )

        # ── CPL Trend Score (30%) ─────────────────────────────────────────
        cpl_values = weekly["CPL (€)"].dropna().tolist()
        cpl_current = cpl_values[-1] if cpl_values else np.nan

        if len(cpl_values) >= 2:
            x = np.arange(len(cpl_values))
            slope = np.polyfit(x, cpl_values, 1)[0]
            cpl_trend_score = max(0, min(100, 50 - slope * 25))
        else:
            cpl_trend_score = 50  # neutral si no hay suficiente historia

        # ── % Alta Intención Score (25%) ──────────────────────────────────
        total_leads = dc["Leads Válidos"].sum()
        total_consid = dc["Consideración"].sum() if "Consideración" in dc.columns else 0
        total_decision = dc["Decisión"].sum() if "Decisión" in dc.columns else 0
        pct_ai = (total_consid + total_decision) / total_leads if total_leads > 0 else 0
        alta_int_score = max(0, min(100, pct_ai * 250))  # 40% → 100

        # ── Coste/Entrevista Score (25%) ──────────────────────────────────
        total_inv = dc["Inversión (€)"].sum()
        total_ent = dc["Entrevistas"].sum()
        coste_ent = total_inv / total_ent if total_ent > 0 else np.nan

        good_ce = b_ce["good"]
        bad_ce = b_ce["bad"]
        if pd.isna(coste_ent) or total_ent == 0:
            coste_ent_score = 0
        elif coste_ent <= good_ce:
            coste_ent_score = 100
        elif coste_ent >= bad_ce:
            coste_ent_score = 0
        else:
            coste_ent_score = 100 * (bad_ce - coste_ent) / (bad_ce - good_ce)

        # ── Tasa de cualificación ─────────────────────────────────────────
        tasa_cual = total_ent / total_leads * 100 if total_leads > 0 else 0

        # ── Volume Score (20%) ────────────────────────────────────────────
        leads_current = dc_current["Leads Válidos"].sum() if not dc_current.empty else 0
        leads_avg = weekly["Leads Válidos"].mean() if not weekly["Leads Válidos"].isna().all() else 0

        if leads_current < 3:
            volume_score = max(0, leads_current / 3 * 30)  # penalización por bajo volumen
        elif leads_avg > 0:
            ratio = leads_current / leads_avg
            volume_score = max(0, min(100, ratio * 50))
        else:
            volume_score = 50

        # ── Health Score Final ────────────────────────────────────────────
        weeks_with_data = len(cpl_values)
        is_new = weeks_with_data < n_weeks

        health = (
            W_CPL_TREND * cpl_trend_score
            + W_ALTA_INT * alta_int_score
            + W_COSTE_ENT * coste_ent_score
            + W_VOLUME * volume_score
        )

        # ── Clasificación ─────────────────────────────────────────────────
        if is_new:
            action = "NUEVA"
        elif health > 70 and cpl_trend_score >= 60:
            action = "ESCALAR"
        elif health >= 40:
            action = "MANTENER"
        elif health >= 25:
            action = "OPTIMIZAR"
        else:
            action = "PAUSAR"

        rows.append({
            "ID_Campaña": camp_id,
            "Programa": programa,
            "Canal": canal,
            "CPL_actual": round(cpl_current, 1) if not pd.isna(cpl_current) else None,
            "CPL_trend": cpl_values,
            "CPL_trend_score": round(cpl_trend_score, 1),
            "PCT_Alta_Intencion": round(pct_ai * 100, 1),
            "Alta_Int_score": round(alta_int_score, 1),
            "Coste_Entrevista": round(coste_ent, 1) if not pd.isna(coste_ent) else None,
            "Coste_Ent_score": round(coste_ent_score, 1),
            "Tasa_Cualificacion": round(tasa_cual, 1),
            "Leads_actual": int(leads_current),
            "Volume_score": round(volume_score, 1),
            "Health_Score": round(health, 1),
            "Action": action,
            "Weeks_Data": weeks_with_data,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("Health_Score", ascending=False).reset_index(drop=True)
    return result


def detect_alerts(
    df_all: pd.DataFrame,
    current_week: int,
) -> list[dict]:
    """Detecta alertas semanales automáticas.

    Returns:
        Lista de dicts con keys: type ('danger'|'warning'|'info'), message, campaign.
    """
    df = df_all[~df_all["Canal"].str.contains("orgánico|seo|organic", case=False, na=False)].copy()
    if df.empty:
        return []

    all_weeks = sorted(df["Semana"].unique())
    if current_week not in all_weeks or len(all_weeks) < 2:
        return []

    idx = all_weeks.index(current_week)
    prev_week = all_weeks[idx - 1] if idx > 0 else None
    if prev_week is None:
        return []

    alerts = []
    campaigns = df["ID_Campaña"].unique()

    for camp in campaigns:
        dc = df[df["ID_Campaña"] == camp]
        curr = dc[dc["Semana"] == current_week]
        prev = dc[dc["Semana"] == prev_week]

        if curr.empty or prev.empty:
            # Campaña con 0 leads esta semana pero tenía antes
            if curr.empty and not prev.empty and prev["Leads Válidos"].sum() > 0:
                alerts.append({
                    "type": "danger",
                    "message": f"**{camp}** tenía {int(prev['Leads Válidos'].sum())} leads en S{prev_week} pero 0 esta semana",
                    "campaign": camp,
                })
            continue

        # CPL spike >30%
        cpl_curr = curr["CPL (€)"].mean()
        cpl_prev = prev["CPL (€)"].mean()
        if not pd.isna(cpl_curr) and not pd.isna(cpl_prev) and cpl_prev > 0:
            pct_change = (cpl_curr - cpl_prev) / cpl_prev * 100
            if pct_change > 30:
                alerts.append({
                    "type": "warning",
                    "message": f"**{camp}**: CPL subió {pct_change:.0f}% ({cpl_prev:.1f}€ → {cpl_curr:.1f}€)",
                    "campaign": camp,
                })

        # Coste/entrevista spike >40%
        inv_curr = curr["Inversión (€)"].sum()
        ent_curr = curr["Entrevistas"].sum()
        inv_prev = prev["Inversión (€)"].sum()
        ent_prev = prev["Entrevistas"].sum()
        if ent_curr > 0 and ent_prev > 0:
            ce_curr = inv_curr / ent_curr
            ce_prev = inv_prev / ent_prev
            if ce_prev > 0:
                ce_change = (ce_curr - ce_prev) / ce_prev * 100
                if ce_change > 40:
                    alerts.append({
                        "type": "warning",
                        "message": f"**{camp}**: Coste/entrevista subió {ce_change:.0f}% ({ce_prev:.0f}€ → {ce_curr:.0f}€)",
                        "campaign": camp,
                    })

    # Ordenar: danger primero, luego warning
    alerts.sort(key=lambda a: 0 if a["type"] == "danger" else 1)
    return alerts


def detect_decline_alerts(
    df_all: pd.DataFrame,
    current_week: int,
    benchmarks: dict,
    health_df: pd.DataFrame,
) -> list[dict]:
    """Detecta campañas veteranas (5+ semanas) que iban bien y ahora declinan.

    Compara el health score actual con el de hace 3 semanas.
    Alerta si pasó de >60 a <45.
    """
    if health_df.empty:
        return []

    all_weeks = sorted(df_all["Semana"].unique())
    if current_week not in all_weeks:
        return []
    idx = all_weeks.index(current_week)
    if idx < 3:
        return []

    past_week = all_weeks[idx - 3]
    past_health = compute_health_score(df_all, past_week, benchmarks)
    if past_health.empty:
        return []

    # Solo campañas veteranas (5+ semanas de datos)
    veteran = health_df[health_df["Weeks_Data"] >= 5]
    if veteran.empty:
        return []

    alerts = []
    for _, row in veteran.iterrows():
        camp = row["ID_Campaña"]
        current_score = row["Health_Score"]
        past_row = past_health[past_health["ID_Campaña"] == camp]
        if past_row.empty:
            continue
        past_score = past_row.iloc[0]["Health_Score"]
        if past_score > 60 and current_score < 45:
            alerts.append({
                "type": "warning",
                "message": (
                    f"📉 **{camp}**: Lleva {row['Weeks_Data']} semanas activa. "
                    f"Health score cayó de {past_score:.0f} → {current_score:.0f}. Revisar tendencia."
                ),
                "campaign": camp,
            })

    return alerts


# Mapeo columna del Sheet → etiqueta legible + diagnóstico
_LOSS_REASONS = {
    "No válido": {
        "label": "No válido",
        "diagnostic": "Datos de contacto erróneos o leads basura. Revisar segmentación y fuentes.",
    },
    "No es lo que buscaba": {
        "label": "No es lo que buscaba",
        "diagnostic": "La campaña atrae público fuera del buyer persona. Revisar audiencia y mensaje.",
    },
    "No tiene dinero": {
        "label": "Precio",
        "diagnostic": "El público no puede asumir el coste. Revisar targeting socioeconómico.",
    },
    "Matriculado en otra escuela": {
        "label": "Competencia",
        "diagnostic": "Leads que se van a competidores. Revisar propuesta de valor y timing.",
    },
    "Ilocalizado": {
        "label": "Ilocalizable",
        "diagnostic": "No se consigue contactar. Revisar calidad del formulario y canales de contacto.",
    },
}


def detect_loss_pattern_alerts(
    df_all: pd.DataFrame,
    current_week: int,
) -> list[dict]:
    """Detecta campañas con motivos de pérdida anormalmente altos vs la media.

    Compara el % de cada motivo de pérdida por campaña con la media global.
    Alerta si una campaña supera la media en más de 15 puntos porcentuales.
    """
    df = df_all[~df_all["Canal"].str.contains("orgánico|seo|organic", case=False, na=False)].copy()
    if df.empty:
        return []

    # Usar datos acumulados (todas las semanas) para patrones más fiables
    # Columnas de motivos disponibles
    available = [col for col in _LOSS_REASONS if col in df.columns]
    if not available:
        return []

    # Agregar por campaña (acumulado todas las semanas)
    agg_cols = {"Perdidos": ("Perdidos", "sum")} if "Perdidos" in df.columns else {}
    for col in available:
        agg_cols[col] = (col, "sum")

    by_camp = df.groupby("ID_Campaña").agg(**agg_cols).reset_index()
    if "Perdidos" not in by_camp.columns:
        by_camp["Perdidos"] = sum(by_camp[col] for col in available)
    by_camp = by_camp[by_camp["Perdidos"] > 0]
    if by_camp.empty:
        return []

    # Calcular % por motivo
    for col in available:
        by_camp[f"pct_{col}"] = by_camp[col] / by_camp["Perdidos"] * 100

    # Media global de cada motivo
    total_perdidos = by_camp["Perdidos"].sum()
    medias = {}
    for col in available:
        medias[col] = by_camp[col].sum() / total_perdidos * 100 if total_perdidos > 0 else 0

    alerts = []
    for _, row in by_camp.iterrows():
        if row["Perdidos"] < 3:  # ignorar campañas con muy pocos perdidos
            continue
        for col in available:
            pct = row[f"pct_{col}"]
            media = medias[col]
            diff = pct - media
            if diff > 10 and pct > 15:  # >10pp por encima de la media Y >15% absoluto
                info = _LOSS_REASONS[col]
                alerts.append({
                    "type": "info",
                    "message": (
                        f"🔍 **{row['ID_Campaña']}**: {info['label']} = {pct:.0f}% "
                        f"(media: {media:.0f}%). {info['diagnostic']}"
                    ),
                    "campaign": row["ID_Campaña"],
                })

    return alerts
