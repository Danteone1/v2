"""
SITER-CAE v5.0 — MASTER MONOLÍTICO CIENTÍFICO (Streamlit)
=========================================================
Laboratorio SAF (Fligstein & McAdam) + Sociofísica + Recursos Limitados
+ Adversario + Bounded Confidence + SPOF + Optimizador + 77 Preguntas
+ Visualización estilo NetLogo + Influencia dirigida con evidencia (E0–E5/SIM)
+ CONFIG externa (YAML/JSON) + BaseModel intercambiable + Experiment.run/validate/save

Ejecutar:
    pip install streamlit pandas numpy pydeck plotly
    (opcional: pip install pyyaml)
    streamlit run app_siter_v50_master.py

MAPA DE MIGRACIÓN AL PAQUETE v3.0 (cada sección → módulo futuro):
    SECCIÓN 0  CONFIG                → configs/*.yaml
    SECCIÓN 1  CORE                  → src/siter_cae/config.py
    SECCIÓN 2  CALIBRADOR            → src/siter_cae/ingestion/
    SECCIÓN 3  LAB                   → src/siter_cae/actors/
    SECCIÓN 4  INDICADORES           → src/siter_cae/validation/
    SECCIÓN 5  MODELOS               → src/siter_cae/sociophysics/ + models/
    SECCIÓN 6  GRAFO DE INFLUENCIA   → src/siter_cae/networks/
    SECCIÓN 7  EXPERIMENTO/MC        → src/siter_cae/simulation/
    SECCIÓN 8  VISUALIZACIÓN         → src/siter_cae/visualization/
    SECCIÓN 9  PREGUNTAS             → src/siter_cae/research/
    SECCIÓN 10 UI                    → app.py (capa delgada)

Sin PII. Datos 100% sintéticos o agregados públicos. Seed+Hash reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
    HAS_PYDECK = True
except ImportError:
    HAS_PYDECK = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# =============================================================================
# SECCIÓN 0 · CONFIG GLOBAL (equivalente monolítico a configs/*.yaml)
# =============================================================================
CONFIG = {
    "model": {"name": "abm_saf", "beta": 1.2, "steps": 15, "seed": 42},
    "abm": {  # antes 'magic numbers' dispersos en el código
        "base_prob": 0.55, "prob_cap": 0.75, "broker_mult": 1.45,
        "evento_mult": 1.15, "evento_duracion": 1, "shock_flip_p": 0.45,
        "umbral_influ": 0.40, "umbral_influ_broker": 0.20, "umbral_skip_p": 0.55,
        "fatiga_gate": 0.80, "fatiga_skip_p": 0.70, "fatiga_add": 0.05,
        "fatiga_recuperacion": 0.95, "cap_pol_decremento": 0.02,
        "costo_min": 10.0, "costo_max": 50.0, "costo_broker": 5.0, "costo_factor": 0.10,
        "eps": 0.40, "eps_rep": 0.80, "mu": 0.30, "repulsion_factor": 0.20,
    },
    "deffuant": {"mu": 0.30, "epsilon": 0.40, "epsilon_rep": 0.80, "repulsion_factor": 0.20},
    "voter": {"resistencia_broker": 0.30},
    "network": {"p_intra": 0.05, "p_inter": 0.01},
    "simulation": {"n_agentes": 400, "mc_runs": 15},
    "habilidades": {"capital_social": 0.30, "acceso_informacion": 0.25,
                    "influencia_liderazgo": 0.25, "arraigo": 0.10,
                    "nivel_movilizacion": 0.10, "desconfianza": -0.15},
    "analisis": {"spof_top_k": 10, "max_paths": 20,
                 "optimizer": {"capitales": [0.70, 0.85, 1.00], "grados": [10, 18, 26],
                               "n_eval": 2, "sub_n": 200, "steps": 8}},
    "opinion_threshold": 0.25,
    "eventos_max_guardados": 3000,
}


def deep_merge(base: dict, override: dict) -> dict:
    """Fusión recursiva: override gana sobre base (config externa sobre defaults)."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def dump_config(cfg: dict) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True) if HAS_YAML else json.dumps(cfg, indent=2, ensure_ascii=False)


# =============================================================================
# SECCIÓN 1 · CORE
# =============================================================================
TERRITORIOS = ["CENTRO", "NORTE", "SUR", "PUEBLA", "EDOMEX", "VERACRUZ"]
TERRITORIOS_COORDS = {
    "CENTRO": (19.4326, -99.1332), "NORTE": (19.5000, -99.1100),
    "SUR": (19.3000, -99.1700), "PUEBLA": (19.0414, -98.2063),
    "EDOMEX": (19.3569, -99.6557), "VERACRUZ": (19.1738, -96.1342),
}
EDGES_TERR = [("CENTRO", "NORTE", 0.8), ("CENTRO", "SUR", 0.7), ("NORTE", "EDOMEX", 0.6),
              ("SUR", "PUEBLA", 0.4), ("CENTRO", "PUEBLA", 0.3)]

COLOR_INTENCION = {"SIMPATIZANTE": [46, 204, 113, 200], "OPOSITOR": [231, 76, 60, 200],
                   "INDECISO": [149, 165, 166, 180]}
COLOR_HEX = {"SIMPATIZANTE": "#2ecc71", "OPOSITOR": "#e74c3c", "INDECISO": "#95a5a6"}
CAMPO_COLOR = {"CONSOLIDACION": "#2ecc71", "DISPUTA_ABIERTA": "#e74c3c", "CONTENCION": "#3498db"}
CAMPO_RGBA = {"CONSOLIDACION": [46, 204, 113, 200], "DISPUTA_ABIERTA": [231, 76, 60, 200],
              "CONTENCION": [52, 152, 219, 200]}
SPIN_MAP = {"SIMPATIZANTE": 1, "OPOSITOR": -1, "INDECISO": 0}


def sha256(v) -> str:
    p = v if isinstance(v, str) else json.dumps(v, sort_keys=True, separators=(",", ":"),
                                                ensure_ascii=False, default=str)
    return hashlib.sha256(p.encode("utf-8")).hexdigest()


def campo_de_territorio(simpat: float, indec: float) -> str:
    if simpat >= 0.5:
        return "CONSOLIDACION"
    if indec >= 0.35:
        return "DISPUTA_ABIERTA"
    return "CONTENCION"


# =============================================================================
# SECCIÓN 2 · CALIBRADOR (ingestión de datos públicos agregados, sin PII)
# =============================================================================
class Calibrador:
    """Importa CSV públicos agregados (ENSU/INE-like) y recalibra distribuciones por territorio."""

    TEMPLATE_COMBINADO = """territorio,simpat_pct,opos_pct,indec_pct,conflicto_pct,inseguridad_pct,desconfianza_proxy,movilizacion_proxy
CENTRO,0.42,0.33,0.25,0.48,0.52,0.38,0.55
NORTE,0.28,0.47,0.25,0.65,0.68,0.55,0.42
SUR,0.38,0.30,0.32,0.41,0.45,0.32,0.68
PUEBLA,0.51,0.28,0.21,0.35,0.40,0.28,0.60
EDOMEX,0.35,0.40,0.25,0.58,0.72,0.50,0.48
VERACRUZ,0.45,0.30,0.25,0.30,0.35,0.25,0.50"""

    @staticmethod
    def parse_csv(uploaded_file_or_text) -> pd.DataFrame:
        if hasattr(uploaded_file_or_text, "read"):
            df = pd.read_csv(uploaded_file_or_text)
        else:
            df = pd.read_csv(StringIO(str(uploaded_file_or_text)))
        df.columns = [c.strip().lower() for c in df.columns]
        if "territorio" not in df.columns:
            raise ValueError("El CSV debe tener columna 'territorio'")
        df["territorio"] = df["territorio"].astype(str).str.upper().str.strip()
        return df

    @staticmethod
    def build_params(df_cal: pd.DataFrame) -> dict:
        params = {}
        for _, row in df_cal.iterrows():
            terr = row["territorio"]
            if terr not in TERRITORIOS:
                continue
            s, o, i = (float(row.get("simpat_pct", 0.40) or 0.40),
                       float(row.get("opos_pct", 0.30) or 0.30),
                       float(row.get("indec_pct", 0.30) or 0.30))
            total = s + o + i
            if total <= 0:
                s, o, i = 0.40, 0.30, 0.30
            else:
                s, o, i = s / total, o / total, i / total
            conflicto = float(row.get("conflicto_pct", 0.45) or 0.45)
            inseg = float(row.get("inseguridad_pct", 0.50) or 0.50)
            desconf = float(row.get("desconfianza_proxy", 0.35) or 0.35)
            mov = float(row.get("movilizacion_proxy", 0.55) or 0.55)
            params[terr] = {
                "p_intencion": [s, o, i],
                "desconf_mean": float(np.clip(desconf, 0.05, 0.9)),
                "exposicion_mean": float(np.clip(0.4 * conflicto + 0.6 * inseg, 0.05, 0.95)),
                "temp_mean": float(np.clip(20 + 70 * np.clip(inseg, 0, 1), 25, 85)),
                "mov_mean": float(np.clip(mov, 0.2, 0.9)),
            }
        return params

    @staticmethod
    def beta_params_from_mean(mean: float, concentration: float = 6.0):
        mean = float(np.clip(mean, 0.05, 0.95))
        return max(0.5, mean * concentration), max(0.5, (1 - mean) * concentration)


# =============================================================================
# SECCIÓN 3 · LAB — generación de universo y broker (actores sintéticos)
# =============================================================================
class Lab:
    """Generación del universo sintético. RNG único (numpy) para reproducibilidad."""

    @staticmethod
    def _habilidades(row: dict, w: dict) -> float:
        return float(np.clip(
            row["capital_social"] * w["capital_social"]
            + row["acceso_informacion"] * w["acceso_informacion"]
            + row["influencia_liderazgo"] * w["influencia_liderazgo"]
            + row["arraigo"] * w["arraigo"]
            + row["nivel_movilizacion"] * w["nivel_movilizacion"]
            + row["desconfianza"] * w["desconfianza"], 0, 1))

    @staticmethod
    def generate(n=400, seed=42, p_intra=0.05, p_inter=0.01, calib_params=None, cfg=None):
        cfg = cfg or CONFIG
        w = cfg["habilidades"]
        rng = np.random.default_rng(seed)
        terr_arr = np.array([t for t in TERRITORIOS
                             if not calib_params or t in calib_params] or TERRITORIOS)
        agents = []
        for i in range(n):
            terr = str(rng.choice(terr_arr))
            base_lat, base_lon = TERRITORIOS_COORDS[terr]
            cp = (calib_params or {}).get(terr, {})
            p_int = cp.get("p_intencion", [0.40, 0.30, 0.30])
            intencion = str(rng.choice(["SIMPATIZANTE", "OPOSITOR", "INDECISO"], p=p_int))
            op_base = 0.7 if intencion == "SIMPATIZANTE" else (-0.7 if intencion == "OPOSITOR" else 0.0)
            a_d, b_d = Calibrador.beta_params_from_mean(cp.get("desconf_mean", 0.28), 5.5)
            a_e, b_e = Calibrador.beta_params_from_mean(cp.get("exposicion_mean", 0.45), 5.0)
            a_m, b_m = Calibrador.beta_params_from_mean(cp.get("mov_mean", 0.65), 6.0)
            agents.append({
                "agent_id": f"SYN-{i:04d}", "territorial_unit_id": terr,
                "lat": float(base_lat + rng.normal(0, 0.025)),
                "lon": float(base_lon + rng.normal(0, 0.025)),
                "capital_social": float(np.clip(rng.beta(3, 3), 0, 1)),
                "acceso_informacion": float(np.clip(rng.beta(3, 2), 0, 1)),
                "influencia_liderazgo": float(np.clip(rng.beta(2, 5), 0, 1)),
                "arraigo": float(np.clip(rng.normal(0.6, 0.2), 0, 1)),
                "nivel_movilizacion": float(np.clip(rng.beta(a_m, b_m), 0, 1)),
                "desconfianza": float(np.clip(rng.beta(a_d, b_d), 0, 1)),
                "exposicion_problema": float(np.clip(rng.beta(a_e, b_e), 0, 1)),
                "intencion": intencion, "spin": SPIN_MAP[intencion],
                "opinion_continua": float(np.clip(op_base + rng.normal(0, 0.18), -1, 1)),
                "resistencia_institucional": float(np.clip(rng.beta(2, 5), 0, 1)),
                "temperatura_sintetica": float(np.clip(rng.normal(cp.get("temp_mean", 55), 12), 20, 90)),
                "fatiga": 0.0, "capital_politico": float(np.clip(rng.uniform(0.5, 1.0), 0, 1)),
                "es_broker_insertado": False, "es_adversario": False,
                "polarizacion_local": float(rng.beta(3, 3)),
                "calibrado": bool(calib_params),
            })
        df = pd.DataFrame(agents)
        df["habilidades_sociales"] = df.apply(lambda r: Lab._habilidades(r.to_dict(), w), axis=1)

        # Red intra + inter-territorio (sobre índices posicionales RangeIndex)
        adj = defaultdict(list)
        for terr in TERRITORIOS:
            idxs = df.index[df["territorial_unit_id"] == terr].tolist()
            for a_ in range(len(idxs)):
                for b_ in range(a_ + 1, len(idxs)):
                    if rng.random() < p_intra:
                        x, y = idxs[a_], idxs[b_]
                        adj[x].append(y)
                        adj[y].append(x)
        for src, tgt, wgt in EDGES_TERR:
            for x in df.index[df["territorial_unit_id"] == src]:
                for y in df.index[df["territorial_unit_id"] == tgt]:
                    if rng.random() < p_inter * wgt:
                        adj[x].append(y)
                        adj[y].append(x)
        df["grado"] = df.index.map(lambda i: len(adj.get(i, [])))
        max_g = int(df["grado"].max() or 1)
        df["influencia_SAF"] = (df["habilidades_sociales"] * 0.6 + (df["grado"] / max_g) * 0.4).clip(0, 1)
        return df, dict(adj)

    @staticmethod
    def insertar_broker(df, adj, territorio, lat, lon, capital, acceso_info, liderazgo,
                        arraigo, mov, desconf, intencion, grado_objetivo=20, seed=42,
                        cfg=None, es_adversario=False):
        """Inserta broker (o adversario) conectado a agentes de alta influencia.
        FIX v5.0: concat con ignore_index=True (el v4.0 duplicaba el índice 0)."""
        cfg = cfg or CONFIG
        rng = np.random.default_rng(seed)
        habilidades = float(np.clip(
            capital * 0.30 + acceso_info * 0.25 + liderazgo * 0.25
            + arraigo * 0.10 + mov * 0.10 - desconf * 0.15, 0, 1))
        pref = "ADV" if es_adversario else "BROKER"
        nuevo = {
            "agent_id": f"{pref}-{uuid.uuid4().hex[:6]}", "territorial_unit_id": territorio,
            "lat": lat, "lon": lon, "capital_social": capital, "acceso_informacion": acceso_info,
            "influencia_liderazgo": liderazgo, "arraigo": arraigo, "nivel_movilizacion": mov,
            "desconfianza": desconf, "exposicion_problema": 0.4, "intencion": intencion,
            "spin": SPIN_MAP[intencion],
            "opinion_continua": 0.75 if intencion == "SIMPATIZANTE"
            else (-0.75 if intencion == "OPOSITOR" else 0.0),
            "resistencia_institucional": 0.3, "temperatura_sintetica": 45.0, "fatiga": 0.0,
            "capital_politico": 0.9, "es_broker_insertado": True, "es_adversario": es_adversario,
            "habilidades_sociales": habilidades, "polarizacion_local": 0.4,
            "grado": 0, "influencia_SAF": 0.0, "calibrado": False,
        }
        df_nuevo = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
        nuevo_idx = len(df_nuevo) - 1  # RangeIndex garantizado

        vecinos_terr = [t for s, t, _ in EDGES_TERR if s == territorio] + \
                       [s for s, t, _ in EDGES_TERR if t == territorio]
        candidatos = df_nuevo.index[df_nuevo["territorial_unit_id"].isin(
            [territorio] + vecinos_terr)].tolist()
        candidatos = [c for c in candidatos if c != nuevo_idx]
        if len(candidatos) < grado_objetivo:
            candidatos = [c for c in df_nuevo.index.tolist() if c != nuevo_idx]
        if candidatos:
            ordenados = sorted(candidatos, key=lambda c: df_nuevo.loc[c, "influencia_SAF"], reverse=True)
            pool = ordenados[:max(grado_objetivo * 2, 1)]
            sel = rng.choice(np.array(pool), size=min(grado_objetivo, len(pool)), replace=False)
            for s in sel:
                adj[nuevo_idx].append(int(s))
                adj[int(s)].append(nuevo_idx)
        df_nuevo["grado"] = df_nuevo.index.map(lambda i: len(adj.get(i, [])))
        max_g = int(df_nuevo["grado"].max() or 1)
        df_nuevo["influencia_SAF"] = (df_nuevo["habilidades_sociales"] * 0.6
                                      + (df_nuevo["grado"] / max_g) * 0.4).clip(0, 1)
        return df_nuevo, adj, nuevo_idx


# =============================================================================
# SECCIÓN 4 · INDICADORES
# =============================================================================
class Indicadores:

    @staticmethod
    def territoriales(df: pd.DataFrame) -> pd.DataFrame:
        res = []
        for terr in df["territorial_unit_id"].unique():
            sub = df[df["territorial_unit_id"] == terr]
            counts = sub["intencion"].value_counts(normalize=True)
            entropy = -sum(p * np.log(p) for p in counts if p > 0)
            simpat = (sub["intencion"] == "SIMPATIZANTE").mean()
            opos = (sub["intencion"] == "OPOSITOR").mean()
            res.append({"territorio": terr, "total": len(sub),
                        "entropia": round(entropy, 3), "herfindahl": round(sum(p ** 2 for p in counts), 3),
                        "polarizacion": round(1 - abs(simpat - opos), 3),
                        "simpat_pct": round(simpat * 100, 1),
                        "mov_prom": round(sub["nivel_movilizacion"].mean(), 3),
                        "habilidades_prom": round(sub["habilidades_sociales"].mean(), 3),
                        "temp_prom": round(sub["temperatura_sintetica"].mean(), 1),
                        "desconf_prom": round(sub["desconfianza"].mean(), 3)})
        return pd.DataFrame(res)

    @staticmethod
    def saf(df: pd.DataFrame) -> pd.DataFrame:
        res = []
        for terr in df["territorial_unit_id"].unique():
            sub = df[df["territorial_unit_id"] == terr]
            counts = sub["intencion"].value_counts(normalize=True)
            entropy = -sum(p * np.log(p) for p in counts if p > 0)
            max_e = np.log(len(counts)) or 1
            estabilidad = 1 - (entropy / max_e)
            simpat = (sub["intencion"] == "SIMPATIZANTE").mean()
            indec = (sub["intencion"] == "INDECISO").mean()
            res.append({"territorio": terr, "campo": campo_de_territorio(simpat, indec),
                        "dominante": counts.idxmax(), "dominancia": round(counts.max(), 3),
                        "estabilidad": round(estabilidad, 3),
                        "conflicto": round(float(sub["spin"].var()) if len(sub) > 1 else 0.0, 3),
                        "fragmentacion": sum(1 for p in counts if p > 0.1),
                        "institucionalizacion": round(estabilidad * 0.6 + (1 - sub["desconfianza"].mean()) * 0.4, 3)})
        return pd.DataFrame(res)

    @staticmethod
    def red(df: pd.DataFrame, adj: dict) -> dict:
        n = len(df)
        aristas = sum(len(v) for v in adj.values()) // 2
        densidad = (2 * aristas) / (n * (n - 1)) if n > 1 else 0
        clust = []
        for i in df.index:
            vec = adj.get(i, [])
            if len(vec) < 2:
                clust.append(0.0)
                continue
            conex = sum(1 for j in range(len(vec)) for k in range(j + 1, len(vec))
                        if vec[k] in adj.get(vec[j], []))
            mc = len(vec) * (len(vec) - 1) / 2
            clust.append(conex / mc if mc > 0 else 0)
        return {"n_nodos": n, "n_aristas": aristas, "densidad": round(densidad, 4),
                "grado_promedio": round(df["grado"].mean(), 2),
                "grado_max": int(df["grado"].max()), "clustering_promedio": round(float(np.mean(clust)), 3)}

    @staticmethod
    def influencia(df: pd.DataFrame) -> dict:
        def gini(x):
            sx = np.sort(x)
            cum = np.cumsum(sx)
            return (len(x) + 1 - 2 * np.sum(cum) / cum[-1]) / len(x) if cum[-1] != 0 else 0
        n = len(df)
        tot = df["influencia_SAF"].sum() or 1
        return {"gini_influencia": round(gini(df["influencia_SAF"].values), 3),
                "top1_share": round(df.nlargest(max(1, int(n * 0.01)), "influencia_SAF")["influencia_SAF"].sum() / tot * 100, 1),
                "top10_share": round(df.nlargest(max(1, int(n * 0.10)), "influencia_SAF")["influencia_SAF"].sum() / tot * 100, 1),
                "broker_ratio": round((df["grado"] > df["grado"].quantile(0.9)).mean(), 3),
                "habilidades_prom": round(df["habilidades_sociales"].mean(), 3),
                "habilidades_top5": round(df.nlargest(max(1, int(n * 0.05)), "influencia_SAF")["habilidades_sociales"].mean(), 3)}


# =============================================================================
# SECCIÓN 5 · MODELOS SOCIOFÍSICOS — BaseModel intercambiable (contrato v3.0)
# =============================================================================
class BaseModel:
    """Contrato común: run(df, adj, steps, seed, ...) → (df_sim, tray, conversiones, events).

    events: lista de InfluenceEvent (dict) con source, target, field, mechanism,
    direction, evidence_level, confidence, step. Sustituible sin tocar el resto.
    """
    name = "base"

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def run(self, df, adj, steps=15, seed=42, evento=None, presupuesto=None):
        raise NotImplementedError


class ABMSAF(BaseModel):
    """Motor compuesto: spins (Voter+shock) + opiniones (Deffuant con repulsión)
    + restricciones (presupuesto, fatiga, capital político) + brokers/adversarios.

    FIXES v5.0 vs v4.0:
    1) El multiplicador del evento SOLO aplica en la ventana del evento (no global).
    2) Actualización 100% síncrona: decisiones con estado inicial del paso;
       efectos en buffers aplicados al cierre del paso (fatiga/capital incluidos).
    3) RNG único (numpy default_rng).
    """
    name = "abm_saf"

    def run(self, df, adj, steps=15, seed=42, evento=None, presupuesto=None):
        p = {**CONFIG["abm"], **self.params}
        rng = np.random.default_rng(seed)
        df_sim = df.reset_index(drop=True).copy()
        ids = df_sim["agent_id"].to_numpy()
        terr = df_sim["territorial_unit_id"].to_numpy()
        spins = df_sim["spin"].to_numpy(dtype=float)
        ops = df_sim["opinion_continua"].to_numpy(dtype=float) if "opinion_continua" in df_sim else None
        fatiga = df_sim["fatiga"].to_numpy(dtype=float) if "fatiga" in df_sim else np.zeros(len(df_sim))
        cap_pol = (df_sim["capital_politico"].to_numpy(dtype=float)
                   if "capital_politico" in df_sim else np.ones(len(df_sim)))
        influ = df_sim["influencia_SAF"].to_numpy(dtype=float)
        brokers = (df_sim["es_broker_insertado"].to_numpy(dtype=bool)
                   if "es_broker_insertado" in df_sim else np.zeros(len(df_sim), bool))
        campo_map = dict(zip(Indicadores.saf(df_sim)["territorio"], Indicadores.saf(df_sim)["campo"]))
        beta = float(p.get("beta", 1.2))
        dinero = float(presupuesto.get("dinero", 999999)) if presupuesto else 999999.0
        horas = float(presupuesto.get("horas", 999999)) if presupuesto else 999999.0
        presupuesto_activo = presupuesto is not None
        conversiones = Counter()
        events = []
        ev0, ev_step, ev_dur = evento or {}, int((evento or {}).get("step", 5)), int(p.get("evento_duracion", 1))
        shock = 0.0
        if evento and evento.get("tipo") == "ESCANDALO":
            shock = -0.5
        elif evento and evento.get("tipo") == "OBRA_PUBLICA":
            shock = 0.5

        def snap(step):
            return {"step": step,
                    "SIMPATIZANTE": float((spins == 1).mean()),
                    "OPOSITOR": float((spins == -1).mean()),
                    "INDECISO": float((spins == 0).mean()),
                    "dinero_restante": dinero, "horas_restante": horas}

        tray = [snap(0)]
        for step in range(1, steps + 1):
            if presupuesto_activo and (dinero <= 0 or horas <= 0):
                break
            new_spins, new_ops = spins.copy(), (ops.copy() if ops is not None else None)
            new_fat, new_cap = fatiga.copy(), cap_pol.copy()
            # FIX 1: ventana del evento con decaimiento, no multiplicador global
            dist_ev = abs(step - ev_step)
            en_ventana = bool(evento) and dist_ev < max(1, ev_dur)
            ev_mult = 1.0 + (float(p["evento_mult"]) - 1.0) * (1 - dist_ev / max(1, ev_dur)) if en_ventana else 1.0
            shock_activo = bool(evento) and step == ev_step
            for i in rng.permutation(len(df_sim)):
                vecinos = adj.get(int(i), [])
                if not vecinos:
                    continue
                if fatiga[i] > p["fatiga_gate"] and rng.random() < p["fatiga_skip_p"]:
                    continue
                umbral = p["umbral_influ_broker"] if brokers[i] else p["umbral_influ"]
                if influ[i] < umbral and rng.random() < p["umbral_skip_p"]:
                    continue
                for v in vecinos:
                    if presupuesto_activo and (dinero <= 0 or horas <= 0):
                        break
                    prob = float(influ[i]) * p["base_prob"] * beta
                    if brokers[i]:
                        prob *= p["broker_mult"]
                    prob *= ev_mult
                    if fatiga[i] > 0.5:
                        prob *= (1 - fatiga[i] * 0.4)
                    if rng.random() < float(np.clip(prob, 0, p["prob_cap"])):
                        # Deffuant con repulsión — decisión con estado inicial del paso
                        if new_ops is not None:
                            diff = abs(ops[i] - ops[v])
                            if diff < p["eps"]:
                                new_ops[v] = float(np.clip(new_ops[v] + p["mu"] * (ops[i] - ops[v]), -1, 1))
                            elif diff > p["eps_rep"]:
                                new_ops[v] = float(np.clip(
                                    new_ops[v] - p["repulsion_factor"] * p["mu"] * (ops[i] - ops[v]), -1, 1))
                        # Spins (Voter + shock)
                        if abs(spins[i] - spins[v]) >= 1 and spins[i] != 0:
                            if shock_activo and shock < 0 and rng.random() < p["shock_flip_p"]:
                                new_spins[v] = -1
                            elif shock_activo and shock > 0 and rng.random() < p["shock_flip_p"]:
                                new_spins[v] = 1
                            else:
                                new_spins[v] = spins[i]
                            conversiones[int(i)] += 1
                            events.append({
                                "source": str(ids[i]), "target": str(ids[v]),
                                "field": campo_map.get(terr[v], "?"),
                                "mechanism": "interaccion_red",
                                "direction": f"{ids[i]}->{ids[v]}",
                                "evidence_level": "SIM",  # evento simulado, no empírico E0–E5
                                "confidence": round(float(np.clip(prob, 0, 1)), 3),
                                "step": step, "territorio": str(terr[v]),
                            })
                            if presupuesto_activo:
                                costo = (float(rng.uniform(p["costo_min"], p["costo_max"]))
                                         if not brokers[i] else p["costo_broker"])
                                dinero -= costo * p["costo_factor"]
                                horas -= 1
                            # FIX 2: fatiga/capital en buffers (síncrono)
                            new_fat[i] = min(1.0, new_fat[i] + p["fatiga_add"])
                            new_cap[i] = max(0.0, new_cap[i] - p["cap_pol_decremento"])
            spins, ops, fatiga, cap_pol = new_spins, new_ops, new_fat, new_cap
            fatiga = np.clip(fatiga * p["fatiga_recuperacion"], 0, 1)
            tray.append(snap(step))
        df_sim["spin"] = spins
        df_sim["intencion"] = np.where(spins == 1, "SIMPATIZANTE", np.where(spins == -1, "OPOSITOR", "INDECISO"))
        if ops is not None:
            df_sim["opinion_continua"] = ops
        df_sim["fatiga"] = fatiga
        df_sim["capital_politico"] = cap_pol
        df_sim["conversiones_causadas"] = [conversiones.get(i, 0) for i in df_sim.index]
        return df_sim, pd.DataFrame(tray), conversiones, events


class VoterSAF(BaseModel):
    """Voter clásico ponderado por influencia SAF. Brokers resisten (semi-zealots)."""
    name = "voter_saf"

    def run(self, df, adj, steps=15, seed=42, evento=None, presupuesto=None):
        p = {**CONFIG["voter"], **self.params}
        beta = float(self.params.get("beta", 1.2))
        rng = np.random.default_rng(seed)
        df_sim = df.reset_index(drop=True).copy()
        ids = df_sim["agent_id"].to_numpy()
        terr = df_sim["territorial_unit_id"].to_numpy()
        spins = df_sim["spin"].to_numpy(dtype=float)
        influ = df_sim["influencia_SAF"].to_numpy(dtype=float)
        brokers = (df_sim["es_broker_insertado"].to_numpy(dtype=bool)
                   if "es_broker_insertado" in df_sim else np.zeros(len(df_sim), bool))
        campo_map = dict(zip(Indicadores.saf(df_sim)["territorio"], Indicadores.saf(df_sim)["campo"]))
        conversiones, events = Counter(), []

        def snap(step):
            return {"step": step, "SIMPATIZANTE": float((spins == 1).mean()),
                    "OPOSITOR": float((spins == -1).mean()), "INDECISO": float((spins == 0).mean())}

        tray = [snap(0)]
        for step in range(1, steps + 1):
            new_spins = spins.copy()
            for i in rng.permutation(len(df_sim)):
                vecinos = adj.get(int(i), [])
                if not vecinos:
                    continue
                v = int(rng.choice(np.array(vecinos)))
                prob = float(np.clip(influ[v] * beta, 0, 0.9))
                if brokers[i]:
                    prob *= p["resistencia_broker"]
                if spins[i] != spins[v] and rng.random() < prob:
                    new_spins[i] = spins[v]
                    conversiones[v] += 1
                    events.append({"source": str(ids[v]), "target": str(ids[i]),
                                   "field": campo_map.get(terr[i], "?"), "mechanism": "voter",
                                   "direction": f"{ids[v]}->{ids[i]}", "evidence_level": "SIM",
                                   "confidence": round(prob, 3), "step": step,
                                   "territorio": str(terr[i])})
            spins = new_spins
            tray.append(snap(step))
        df_sim["spin"] = spins
        df_sim["intencion"] = np.where(spins == 1, "SIMPATIZANTE", np.where(spins == -1, "OPOSITOR", "INDECISO"))
        df_sim["conversiones_causadas"] = [conversiones.get(i, 0) for i in df_sim.index]
        return df_sim, pd.DataFrame(tray), conversiones, events


class DeffuantSAF(BaseModel):
    """Deffuant bounded confidence + repulsión sobre opiniones continuas.
    La intención discreta se deriva de la opinión final (umbral ±0.25)."""
    name = "deffuant_saf"

    def run(self, df, adj, steps=15, seed=42, evento=None, presupuesto=None):
        p = {**CONFIG["deffuant"], **self.params}
        rng = np.random.default_rng(seed)
        df_sim = df.reset_index(drop=True).copy()
        ids = df_sim["agent_id"].to_numpy()
        terr = df_sim["territorial_unit_id"].to_numpy()
        ops = df_sim["opinion_continua"].to_numpy(dtype=float)
        campo_map = dict(zip(Indicadores.saf(df_sim)["territorio"], Indicadores.saf(df_sim)["campo"]))
        events = []
        thr = CONFIG["opinion_threshold"]

        def snap(step):
            return {"step": step, "SIMPATIZANTE": float((ops > thr).mean()),
                    "OPOSITOR": float((ops < -thr).mean()), "INDECISO": float((np.abs(ops) <= thr).mean())}

        tray = [snap(0)]
        for step in range(1, steps + 1):
            new_ops = ops.copy()
            for i in rng.permutation(len(df_sim)):
                vecinos = adj.get(int(i), [])
                if not vecinos:
                    continue
                v = int(rng.choice(np.array(vecinos)))
                d = abs(ops[i] - ops[v])
                if d < p["epsilon"]:
                    delta = p["mu"] / 2 * (ops[v] - ops[i])
                    new_ops[i], new_ops[v] = np.clip(new_ops[i] + delta, -1, 1), np.clip(new_ops[v] - delta, -1, 1)
                    mech = "deffuant_convergencia"
                elif d > p["epsilon_rep"]:
                    delta = p["repulsion_factor"] * p["mu"] / 2 * (ops[v] - ops[i])
                    new_ops[i], new_ops[v] = np.clip(new_ops[i] - delta, -1, 1), np.clip(new_ops[v] + delta, -1, 1)
                    mech = "deffuant_repulsion"
                else:
                    continue
                if abs(delta) > 0.005:
                    events.append({"source": str(ids[i]), "target": str(ids[v]), "field": campo_map.get(terr[v], "?"),
                                   "mechanism": mech, "direction": f"{ids[i]}->{ids[v]}",
                                   "evidence_level": "SIM", "confidence": round(abs(delta) * 10, 3),
                                   "step": step, "territorio": str(terr[v])})
            ops = new_ops
            tray.append(snap(step))
        df_sim["opinion_continua"] = ops
        df_sim["spin"] = np.where(ops > thr, 1, np.where(ops < -thr, -1, 0))
        df_sim["intencion"] = np.where(ops > thr, "SIMPATIZANTE", np.where(ops < -thr, "OPOSITOR", "INDECISO"))
        return df_sim, pd.DataFrame(tray), Counter(), events


MODEL_REGISTRY = {"abm_saf": ABMSAF, "voter_saf": VoterSAF, "deffuant_saf": DeffuantSAF}


# =============================================================================
# SECCIÓN 6 · GRAFO DE INFLUENCIA — InfluenceEvent + FieldGraph (v3.0)
# =============================================================================
EVIDENCIA_DESC = {
    "E0": "sin evidencia", "E1": "coocurrencia", "E2": "relación documental",
    "E3": "relación institucional observable", "E4": "relación repetida temporalmente",
    "E5": "evidencia de mecanismo causal",
    "SIM": "evento simulado (motor sociofísico, NO evidencia empírica)",
}


class FieldGraph:
    """Grafo multicapa ligero: actores + campos (territorio) + eventos de influencia.

    Métodos clave (contrato v3.0):
      influence_sources(target, field)  → ¿quién influye sobre X?
      influence_paths(source, target, max_hops) → rutas actor-actor y campo-mediadas
    """

    def __init__(self, df: pd.DataFrame, adj: dict, events: list):
        self.df = df.reset_index(drop=True)
        self.adj = adj
        self.events = events
        self.ids = self.df["agent_id"].tolist()
        self.terr = self.df["territorial_unit_id"].tolist()
        self.idx_of = {aid: i for i, aid in enumerate(self.ids)}

    def influence_sources(self, target: str, field: str | None = None) -> pd.DataFrame:
        rows = [e for e in self.events if e["target"] == target
                and (field is None or e["field"] == field or e["territorio"] == field)]
        df_ev = pd.DataFrame(rows)
        # Fuentes potenciales (estructura de red, etiquetadas como tales — nunca causalidad)
        if target in self.idx_of:
            i = self.idx_of[target]
            pot = []
            for v in self.adj.get(i, []):
                pot.append({"source": self.ids[v], "target": target, "field": "?",
                            "mechanism": "estructura_red", "direction": f"{self.ids[v]}->{target}",
                            "evidence_level": "E2", "confidence": round(float(self.df['influencia_SAF'].iloc[v]), 3),
                            "step": -1, "territorio": self.terr[v]})
            df_pot = pd.DataFrame(pot)
        else:
            df_pot = pd.DataFrame()
        return df_ev, df_pot

    def _mixed_graph(self) -> dict:
        g = defaultdict(set)
        for i, nbrs in self.adj.items():
            if 0 <= int(i) < len(self.ids):
                u = self.ids[int(i)]
                for v in nbrs:
                    if 0 <= int(v) < len(self.ids):
                        g[u].add(self.ids[int(v)])
                        g[self.ids[int(v)]].add(u)
        for pos, aid in enumerate(self.ids):
            f = f"F:{self.terr[pos]}"
            g[aid].add(f)
            g[f].add(aid)
        for s, t, _ in EDGES_TERR:
            g[f"F:{s}"].add(f"F:{t}")
            g[f"F:{t}"].add(f"F:{s}")
        return g

    def influence_paths(self, source: str, target: str, max_hops: int = 3) -> list:
        g = self._mixed_graph()
        results = []

        def dfs(node, path, visited):
            if len(results) >= CONFIG["analisis"]["max_paths"]:
                return
            if node == target:
                hops = len(path) - 1
                if 0 < hops <= max_hops + 2:
                    tipo = ("campo-mediada" if any(str(n).startswith("F:") for n in path[1:-1])
                            else "actor-actor")
                    results.append({"path": path, "hops": hops, "tipo": tipo})
                return
            if len(path) > max_hops + 2:
                return
            for nb in g.get(node, []):
                if nb not in visited:
                    dfs(nb, path + [nb], visited | {nb})

        dfs(source, [source], {source})
        results.sort(key=lambda r: r["hops"])
        return results[:CONFIG["analisis"]["max_paths"]]

    def matriz(self) -> pd.DataFrame:
        if not self.events:
            return pd.DataFrame()
        df = pd.DataFrame(self.events)
        agg = (df.groupby(["source", "target"])
               .agg(n_eventos=("step", "count"), confianza_media=("confidence", "mean"),
                    pasos=("step", lambda s: f"{s.min()}-{s.max()}"),
                    campos=("field", lambda x: ",".join(sorted(set(x)))),
                    mecanismos=("mechanism", lambda x: ",".join(sorted(set(x)))))
               .reset_index())
        return agg.sort_values("n_eventos", ascending=False)


def componentes(adj: dict, nodes: list) -> list:
    seen, comps = set(), []
    for n0 in nodes:
        if n0 in seen:
            continue
        comp, stack = [], [n0]
        seen.add(n0)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(comp)
    return comps


def spof_analisis(df: pd.DataFrame, adj: dict, top_k: int = 10) -> pd.DataFrame:
    """SPOF: actores cuya remoción fragmenta la red (punto único de falla del campo)."""
    nodes = list(df.index)
    base = componentes(adj, nodes)
    base_n = len(base)
    base_giant = max(len(c) for c in base) / len(nodes) if nodes else 0
    top = df.nlargest(top_k, "influencia_SAF")
    rows = []
    for i in top.index:
        adj2 = {k: [x for x in v if x != i] for k, v in adj.items() if k != i}
        comps = componentes(adj2, [n for n in nodes if n != i])
        giant = max(len(c) for c in comps) / max(1, len(nodes) - 1)
        rows.append({"agent_id": df.loc[i, "agent_id"], "territorio": df.loc[i, "territorial_unit_id"],
                     "influencia_SAF": round(float(df.loc[i, "influencia_SAF"]), 3),
                     "grado": int(df.loc[i, "grado"]),
                     "componentes_antes": base_n, "componentes_despues": len(comps),
                     "delta_fragmentacion": round(len(comps) - base_n + (base_giant - giant), 3),
                     "giant_despues": round(giant, 3)})
    return pd.DataFrame(rows).sort_values("delta_fragmentacion", ascending=False)


# =============================================================================
# SECCIÓN 7 · EXPERIMENTO, MONTE CARLO, OPTIMIZADOR
# =============================================================================
def run_montecarlo(df, adj, engine="abm_saf", steps=15, n_sim=15, seed=42,
                   beta=1.2, evento=None, presupuesto=None, params=None, progress=None):
    model_cls = MODEL_REGISTRY.get(engine, ABMSAF)
    model = model_cls({**(params or {}), "beta": beta})
    trays, finales, campos_rows = [], [], []
    for k in range(n_sim):
        if progress:
            progress((k + 1) / n_sim)
        df_fin, tray, _, _ = model.run(df, adj, steps=steps, seed=seed + 1000 * (k + 1),
                                       evento=evento, presupuesto=presupuesto)
        tray = tray.copy()
        tray["run"] = k
        trays.append(tray)
        fin = tray.iloc[-1]
        finales.append({"run": k, "SIMPATIZANTE": fin["SIMPATIZANTE"],
                        "OPOSITOR": fin["OPOSITOR"], "INDECISO": fin["INDECISO"]})
        for terr in df_fin["territorial_unit_id"].unique():
            sub = df_fin[df_fin["territorial_unit_id"] == terr]
            simpat = (sub["intencion"] == "SIMPATIZANTE").mean()
            indec = (sub["intencion"] == "INDECISO").mean()
            campos_rows.append({"run": k, "territorio": terr, "campo": campo_de_territorio(simpat, indec)})
    runs_df = pd.concat(trays, ignore_index=True)
    g = runs_df.groupby("step")
    summary = pd.DataFrame({"step": list(g.groups.keys())})
    for col in ["SIMPATIZANTE", "OPOSITOR", "INDECISO"]:
        summary[f"{col}_mean"] = g[col].mean().values
        summary[f"{col}_lo"] = g[col].quantile(0.05).values
        summary[f"{col}_hi"] = g[col].quantile(0.95).values
    campos_prob = (pd.DataFrame(campos_rows).groupby(["territorio", "campo"]).size()
                   .groupby(level=0).apply(lambda s: s / s.sum()).reset_index(name="prob"))
    return {"engine": engine, "n_sim": n_sim, "runs": runs_df, "summary": summary,
            "finales": pd.DataFrame(finales), "campos_prob": campos_prob,
            "seed": seed, "steps": steps, "beta": beta}


class Experiment:
    """Patrón v3.0: run() → validate() → save(). Reproducibilidad académica."""

    def __init__(self, name: str, engine: str, config: dict, seed: int,
                 dataset_version: str = "SYNTHETIC-v5.0"):
        self.name, self.engine, self.config, self.seed = name, engine, config, seed
        self.dataset_version = dataset_version
        self.df_sim, self.tray, self.events, self.checks = None, None, [], []

    def run(self, df, adj, steps=15, beta=1.2, evento=None, presupuesto=None):
        model = MODEL_REGISTRY.get(self.engine, ABMSAF)({**self.config.get("abm", {}),
                                                         "beta": beta})
        self.df_sim, self.tray, _, ev = model.run(df, adj, steps=steps, seed=self.seed,
                                                  evento=evento, presupuesto=presupuesto)
        self.events = ev[:CONFIG["eventos_max_guardados"]]
        self.n_events_total = len(ev)
        return self

    def validate(self) -> list:
        checks = []
        if self.tray is None:
            return [{"check": "trayectoria", "ok": False, "detalle": "sin ejecución"}]
        shares = self.tray[["SIMPATIZANTE", "OPOSITOR", "INDECISO"]].sum(axis=1)
        checks.append({"check": "tray: sumas=1", "ok": bool(np.allclose(shares, 1.0, atol=1e-6)),
                       "detalle": f"max desviación={float((shares-1).abs().max()):.2e}"})
        ok_nan = bool(self.df_sim[["spin"] + (["opinion_continua"] if "opinion_continua" in self.df_sim else [])]
                      .notna().all().all())
        checks.append({"check": "sin NaN en estado", "ok": ok_nan, "detalle": "spins/opiniones"})
        confs = [e["confidence"] for e in self.events]
        checks.append({"check": "confianza ∈ [0,1]", "ok": all(0 <= c <= 1 for c in confs) if confs else True,
                       "detalle": f"n_eventos_guardados={len(self.events)}/{self.n_events_total}"})
        ev_ok = all(e["evidence_level"] in EVIDENCIA_DESC for e in self.events)
        checks.append({"check": "evidencia etiquetada (E0–E5/SIM)", "ok": ev_ok,
                       "detalle": "sin causalidad empírica afirmada"})
        self.checks = checks
        return checks

    def payload(self, df_terr, df_saf, red_ind, inf_ind, extra=None):
        data = {
            "metadata": {"experiment_id": f"{self.name}-{uuid.uuid4().hex[:4]}",
                         "dataset_version": self.dataset_version, "engine": self.engine,
                         "model_version": "siter-v5.0-master", "seed": self.seed,
                         "config": self.config, "timestamp": pd.Timestamp.now().isoformat(),
                         "data_origin": "CALCULATED_FROM_SYNTHETIC", "validation": self.checks},
            "territorial_fields": (df_terr.merge(df_saf, on="territorio", how="left")
                                   .assign(data_origin="CALCULATED_FROM_SYNTHETIC")
                                   .to_dict(orient="records")),
            "indicadores": {"red": red_ind, "influencia": inf_ind,
                            "trayectoria": self.tray.to_dict(orient="records")},
            "influence_events": {"n_total": self.n_events_total, "guardados": self.events},
            "governance": {"public": True, "personal": False, "agregado": True},
        }
        if extra:
            data["extra"] = extra
        h = sha256(data)  # FIX v5.0: hash del payload COMPLETO, no solo territorial_fields
        data["metadata"]["output_hash"] = h
        data["governance"]["hash_integridad"] = h
        return data

    def save(self, payload: dict, path="outputs") -> str:
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, f"{self.name}_{payload['metadata']['experiment_id']}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, default=str, indent=2)
        return fname


def subuniverso(df, adj, n_max):
    if len(df) <= n_max:
        return df.copy(), {k: list(v) for k, v in adj.items()}
    per = max(5, n_max // len(df["territorial_unit_id"].unique()))
    keep = df.groupby("territorial_unit_id").head(per).index.tolist()
    keep_set = set(keep)
    adj_sub = {int(k): [int(x) for x in v if x in keep_set] for k, v in adj.items() if int(k) in keep_set}
    return df.loc[keep].reset_index(drop=True), adj_sub


def optimizar_broker(df, adj, intencion="SIMPATIZANTE", cfg=None, beta=1.2, progress=None):
    """Búsqueda en grilla: territorio × capital × grado → mejor inserción de broker."""
    cfg = cfg or CONFIG
    opt = cfg["analisis"]["optimizer"]
    df_s, adj_s = subuniverso(df, adj, opt["sub_n"])
    capitales, grados, n_eval, steps = opt["capitales"], opt["grados"], opt["n_eval"], opt["steps"]
    model = ABMSAF({**cfg["abm"], "beta": beta})
    # Línea base sin broker
    base_scores = []
    for e in range(n_eval):
        _, tray_b, _, _ = model.run(df_s, adj_s, steps=steps, seed=4242 + e)
        base_scores.append(float(tray_b.iloc[-1][intencion]))
    base = float(np.mean(base_scores))
    combos = [(t, c, g) for t in TERRITORIOS for c in capitales for g in grados]
    rows = []
    for k, (t, c, g) in enumerate(combos):
        if progress:
            progress(k / len(combos))
        scores = []
        for e in range(n_eval):
            sd = 9000 + 101 * e + 7 * k
            dfb, adjb, _ = Lab.insertar_broker(
                df_s, {kk: list(vv) for kk, vv in adj_s.items()},
                territorio=t, lat=TERRITORIOS_COORDS[t][0], lon=TERRITORIOS_COORDS[t][1],
                capital=c, acceso_info=c, liderazgo=c, arraigo=0.7, mov=0.75, desconf=0.15,
                intencion=intencion, grado_objetivo=g, seed=sd, cfg=cfg)
            _, tray, _, _ = model.run(dfb, adjb, steps=steps, seed=sd)
            scores.append(float(tray.iloc[-1][intencion]))
        rows.append({"territorio": t, "capital": c, "grado": g,
                     "score": round(float(np.mean(scores)), 4),
                     "delta_vs_base": round(float(np.mean(scores)) - base, 4)})
    if progress:
        progress(1.0)
    return pd.DataFrame(rows).sort_values("score", ascending=False), base


# =============================================================================
# SECCIÓN 8 · VISUAL MAPS (rng único)
# =============================================================================
class VisualMaps:

    @staticmethod
    def generar_rutas(n_brigadas=4, pasos=8, seed=42):
        rng = np.random.default_rng(seed)
        terr_arr = np.array(TERRITORIOS)
        rutas = []
        for b in range(n_brigadas):
            terr = str(rng.choice(terr_arr))
            lat, lon = TERRITORIOS_COORDS[terr]
            ruta = [{"brigada": f"B-{b}", "step": 0, "lat": lat, "lon": lon, "territorio": terr}]
            for step in range(1, pasos + 1):
                vecinos = [t for s, t, _ in EDGES_TERR if s == ruta[-1]["territorio"]] + \
                          [s for s, t, _ in EDGES_TERR if t == ruta[-1]["territorio"]]
                nuevo_terr = str(rng.choice(np.array(vecinos))) if vecinos and rng.random() < 0.7 \
                    else str(rng.choice(terr_arr))
                bl, bo = TERRITORIOS_COORDS[nuevo_terr]
                ruta.append({"brigada": f"B-{b}", "step": step,
                             "lat": bl + rng.normal(0, 0.01), "lon": bo + rng.normal(0, 0.01),
                             "territorio": nuevo_terr})
            rutas.extend(ruta)
        return pd.DataFrame(rutas)

    @staticmethod
    def generar_ondas(origen="CENTRO", pasos=10, seed=42):
        rng = np.random.default_rng(seed)
        bl, bo = TERRITORIOS_COORDS[origen]
        ondas = []
        for step in range(pasos + 1):
            radio_km = step * 5
            intensidad = max(0.0, 100 - step * 8 + rng.normal(0, 5))
            for ang in np.linspace(0, 2 * np.pi, 18):
                d_lat = (radio_km / 111) * np.cos(ang)
                d_lon = (radio_km / (111 * np.cos(np.radians(bl)))) * np.sin(ang)
                intens = max(0.0, intensidad)
                ondas.append({"origen": origen, "step": step, "radio_km": radio_km,
                              "intensidad": intens, "lat": bl + d_lat + rng.normal(0, 0.008),
                              "lon": bo + d_lon + rng.normal(0, 0.008),
                              "color_rgba": [255, int(120 + 135 * (1 - intens / 100)), 30, 150]})
        return pd.DataFrame(ondas)


# =============================================================================
# SECCIÓN 9 · 77 PREGUNTAS (respuestas CALCULADAS, no inventadas)
# =============================================================================
def build_preguntas(df, df_terr, df_saf, red_ind, inf_ind) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["n", "categoria", "pregunta", "respuesta", "fuente", "evidencia"])
    filas, n = [], [0]

    def add(cat, preg, resp, fuente):
        n[0] += 1
        filas.append({"n": n[0], "categoria": cat, "pregunta": preg, "respuesta": str(resp),
                      "fuente": fuente, "evidencia": "SIM-CALC (sintético/calibrado)"})

    t = df_terr.set_index("territorio")
    s = df_saf.set_index("territorio")
    add("Global", "¿Cuántos agentes sintéticos componen el universo?", len(df), "len(df)")
    add("Global", "¿Cuántos territorios participan?", df["territorial_unit_id"].nunique(), "df")
    for met, nombre, fmt in [("simpat_pct", "mayor simpatía", "{:.1f}%"), ("polarizacion", "mayor polarización", "{:.3f}"),
                             ("entropia", "mayor entropía", "{:.3f}"), ("temp_prom", "mayor temperatura/enojo", "{:.1f}"),
                             ("desconf_prom", "mayor desconfianza", "{:.3f}"), ("mov_prom", "mayor movilización", "{:.3f}"),
                             ("habilidades_prom", "mayores habilidades sociales", "{:.3f}")]:
        if met in t.columns:
            add("Global", f"¿Qué territorio tiene {nombre}?", f"{t[met].idxmax()} ({fmt.format(t[met].max())})", "Indicadores.territoriales")
    add("Global", "¿Qué campo SAF domina (más territorios)?",
        f"{df_saf['campo'].value_counts().idxmax()} ({df_saf['campo'].value_counts().max()} territorios)", "Indicadores.saf")
    for met, nombre in [("estabilidad", "más estable"), ("conflicto", "más conflicto"),
                        ("institucionalizacion", "más institucionalizado"), ("fragmentacion", "más fragmentado")]:
        if met in s.columns:
            add("Global", f"¿Qué territorio- campo está {nombre}?",
                f"{s[met].idxmax()} ({float(s[met].max()):.3f})", "Indicadores.saf")
    add("Global", "¿Agente más influyente (id, valor)?",
        f"{df.loc[df['influencia_SAF'].idxmax(), 'agent_id']} ({df['influencia_SAF'].max():.3f})", "df.influencia_SAF")
    add("Global", "¿Gini de influencia?", inf_ind.get("gini_influencia"), "Indicadores.influencia")
    # Bloque por territorio (6 × 8 = 48)
    for terr in t.index:
        r, q = t.loc[terr], s.loc[terr]
        add(f"Territorio {terr}", f"[{terr}] ¿Campo de acción estratégica?", r.get("campo", q.get("campo")), "Indicadores.saf")
        add(f"Territorio {terr}", f"[{terr}] ¿Rol dominante?", q.get("dominante"), "Indicadores.saf")
        add(f"Territorio {terr}", f"[{terr}] ¿Dominancia?", q.get("dominancia"), "Indicadores.saf")
        add(f"Territorio {terr}", f"[{terr}] ¿Estabilidad?", q.get("estabilidad"), "Indicadores.saf")
        add(f"Territorio {terr}", f"[{terr}] ¿Conflicto (varianza spin)?", q.get("conflicto"), "Indicadores.saf")
        add(f"Territorio {terr}", f"[{terr}] ¿Fragmentación?", q.get("fragmentacion"), "Indicadores.saf")
        add(f"Territorio {terr}", f"[{terr}] ¿Institucionalización?", q.get("institucionalizacion"), "Indicadores.saf")
        add(f"Territorio {terr}", f"[{terr}] ¿Simpatía (%)?", r.get("simpat_pct"), "Indicadores.territoriales")
    # Bloque de red (10 → total 77)
    for k, v in list(red_ind.items())[:6]:
        add("Red", f"¿{k.replace('_', ' ')}?", v, "Indicadores.red")
    for k, v in list(inf_ind.items()):
        if k not in ("gini_influencia",) and len([f for f in filas if f["fuente"] == "Indicadores.influencia"]) < 4:
            add("Red", f"¿{k.replace('_', ' ')} (influencia)?", v, "Indicadores.influencia")
    while len(filas) < 77:
        top = df.nlargest(77 - len(filas), "influencia_SAF")
        for _, r in top.iterrows():
            add("Actores", f"¿Influencia de {r['agent_id']} ({r['territorial_unit_id']})?",
                round(float(r["influencia_SAF"]), 3), "df.influencia_SAF")
            if len(filas) >= 77:
                break
    return pd.DataFrame(filas[:77])


# =============================================================================
# SECCIÓN 10 · UI STREAMLIT
# =============================================================================
st.set_page_config(page_title="SITER-CAE v5.0 MASTER", page_icon="🧠", layout="wide")

st.title("🧠 SITER-CAE v5.0 — MASTER MONOLÍTICO CIENTÍFICO")
st.caption("SAF + Sociofísica + Recursos Limitados + Adversario + Bounded Confidence + SPOF + "
           "Optimizador + 77 Preguntas + Influencia dirigida (E0–E5/SIM) + BaseModel intercambiable "
           "+ Experiment.run/validate/save | Sin PII | Seed+Hash reproducible")

if "s5" not in st.session_state:
    st.session_state.s5 = {"df": pd.DataFrame(), "adj": {}, "df_base": pd.DataFrame(), "adj_base": {},
                           "df_terr": pd.DataFrame(), "df_saf": pd.DataFrame(), "red_ind": {}, "inf_ind": {},
                           "tray": pd.DataFrame(), "df_final": pd.DataFrame(), "events": [],
                           "n_events_total": 0, "rutas": pd.DataFrame(), "ondas": pd.DataFrame(),
                           "mc": None, "calib_params": None, "calib_source": None, "calib_hash": None,
                           "config": CONFIG, "broker_ids": [], "adv_ids": []}

S = st.session_state.s5

# ----- SIDEBAR -----
st.sidebar.header("⚙️ Universo Base")
seed = st.sidebar.number_input("Seed", 1, 99999, int(CONFIG["model"]["seed"]))
n = st.sidebar.slider("N agentes", 100, 1500, CONFIG["simulation"]["n_agentes"], 50)
p_intra = st.sidebar.slider("Prob intra-territorio", 0.01, 0.15, CONFIG["network"]["p_intra"], 0.01)
p_inter = st.sidebar.slider("Prob inter-territorio", 0.001, 0.08, CONFIG["network"]["p_inter"], 0.001)
beta = st.sidebar.slider("Beta influencia", 0.3, 2.5, float(CONFIG["model"]["beta"]), 0.1)
steps = st.sidebar.slider("Pasos ABM", 5, 40, int(CONFIG["model"]["steps"]))
n_sim = st.sidebar.slider("N sim Monte Carlo", 5, 40, CONFIG["simulation"]["mc_runs"])
usar_calib = st.sidebar.checkbox("Usar calibración con datos públicos", value=bool(S["calib_params"]))

with st.sidebar.expander("🧾 Config externa (YAML/JSON)"):
    st.caption("Parámetros fuera del código (equivalente a configs/*.yaml). Los sliders de la izquierda "
               "sobrescriben en runtime; la config queda registrada en el export.")
    cfg_text = st.text_area("config", value=dump_config(S["config"]), height=220, key="cfg_text")
    if st.button("Aplicar config"):
        try:
            parsed = yaml.safe_load(cfg_text) if HAS_YAML else json.loads(cfg_text)
            S["config"] = deep_merge(CONFIG, parsed or {})
            st.success("Config aplicada. Regenera universo / corre ABM para usarla.")
        except Exception as e:
            st.error(f"Config inválida: {e}")

if st.sidebar.button("🧬 Generar universo", type="primary"):
    calib = S.get("calib_params") if usar_calib else None
    df, adj = Lab.generate(n, seed=seed, p_intra=p_intra, p_inter=p_inter,
                           calib_params=calib, cfg=S["config"])
    S.update({"df": df, "adj": adj, "df_base": df.copy(), "adj_base": {k: list(v) for k, v in adj.items()},
              "tray": pd.DataFrame(), "df_final": pd.DataFrame(), "events": [], "n_events_total": 0,
              "mc": None, "broker_ids": [], "adv_ids": []})
    st.success(f"Universo: {len(df)} agentes | {'CALIBRADO' if calib else 'SINTÉTICO puro'} | seed={seed}")

if st.sidebar.button("♻️ Restaurar universo base"):
    S["df"] = S["df_base"].copy()
    S["adj"] = {k: list(v) for k, v in S["adj_base"].items()}
    S.update({"tray": pd.DataFrame(), "df_final": pd.DataFrame(), "events": [], "n_events_total": 0,
              "mc": None, "broker_ids": [], "adv_ids": []})
    st.success("Universo restaurado (brokers/adversarios eliminados)")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Broker SAF (tab 5)")
territorio_broker = st.sidebar.selectbox("Territorio", TERRITORIOS, index=0)
lat_b = st.sidebar.number_input("Lat", value=TERRITORIOS_COORDS[territorio_broker][0], format="%.4f")
lon_b = st.sidebar.number_input("Lon", value=TERRITORIOS_COORDS[territorio_broker][1], format="%.4f")
intencion_b = st.sidebar.selectbox("Intención", ["SIMPATIZANTE", "OPOSITOR", "INDECISO"])
grado_b = st.sidebar.slider("Grado objetivo", 5, 40, 18)
capital = st.sidebar.slider("Capital social", 0.0, 1.0, 0.85, 0.05)
acceso = st.sidebar.slider("Acceso información", 0.0, 1.0, 0.80, 0.05)
lider = st.sidebar.slider("Liderazgo", 0.0, 1.0, 0.85, 0.05)
arraigo = st.sidebar.slider("Arraigo", 0.0, 1.0, 0.70, 0.05)
mov = st.sidebar.slider("Movilización", 0.0, 1.0, 0.75, 0.05)
desconf = st.sidebar.slider("Desconfianza", 0.0, 1.0, 0.15, 0.05)
habil = max(0.0, min(1.0, capital * 0.30 + acceso * 0.25 + lider * 0.25 + arraigo * 0.10 + mov * 0.10 - desconf * 0.15))
st.sidebar.metric("Habilidades SAF", f"{habil:.2f}")

df, adj = S["df"], S["adj"]

tabs = st.tabs(["1️⃣ Indicadores & SAF", "2️⃣ ABM Dinámica", "3️⃣ 🗺️ Visual Maps",
                "4️⃣ 🔮 Monte Carlo", "5️⃣ 🧠 Broker & Duelo", "6️⃣ 🕸️ Influencia & Grafo",
                "7️⃣ 📋 77 Preguntas", "8️⃣ 📤 Experimento & Export", "9️⃣ 📥 Calibración"])

# ---------- TAB 1 ----------
with tabs[0]:
    if df.empty:
        st.info("➡️ Genera el universo en la barra lateral")
    else:
        df_terr, df_saf = Indicadores.territoriales(df), Indicadores.saf(df)
        red_ind, inf_ind = Indicadores.red(df, adj), Indicadores.influencia(df)
        S.update({"df_terr": df_terr, "df_saf": df_saf, "red_ind": red_ind, "inf_ind": inf_ind})
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Territoriales")
            st.dataframe(df_terr, use_container_width=True)
            if HAS_PLOTLY:
                st.plotly_chart(px.bar(df_terr, x="territorio", y="simpat_pct", color="polarizacion",
                                       title="Simpatía % y Polarización", color_continuous_scale="RdYlGn_r"),
                                use_container_width=True)
        with c2:
            st.subheader("Campos SAF")
            st.dataframe(df_saf, use_container_width=True)
            st.bar_chart(df_saf["campo"].value_counts())
        st.subheader("Red e Influencia")
        st.json({"red": red_ind, "influencia": inf_ind})

# ---------- TAB 2 ----------
with tabs[1]:
    if df.empty:
        st.info("➡️ Genera el universo primero")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            engine = st.selectbox("Motor (BaseModel intercambiable)",
                                  ["abm_saf", "voter_saf", "deffuant_saf"],
                                  help="Mismos datos, distintos mecanismos — comparable sin reconstruir nada.")
            evento_tipo = st.selectbox("Evento", ["Ninguno", "ESCANDALO", "OBRA_PUBLICA"])
            evento_step = st.number_input("Paso del evento", 1, steps, 3)
            evento_dur = st.number_input("Duración evento (pasos, con decaimiento)", 1, 5, 1,
                                         help="FIX v5.0: el multiplicador solo aplica dentro de esta ventana.")
            usar_presupuesto = st.checkbox("Activar presupuesto limitado", value=False)
            dinero_p = st.number_input("Dinero $", 500, 20000, 5000, 500) if usar_presupuesto else 999999
            horas_p = st.number_input("Horas", 20, 500, 100, 10) if usar_presupuesto else 999999
        with col2:
            if st.button("▶️ Correr simulación", type="primary"):
                evento = None
                if evento_tipo != "Ninguno":
                    S["config"] = deep_merge(S["config"], {"abm": {"evento_duracion": int(evento_dur)}})
                    evento = {"tipo": evento_tipo, "step": int(evento_step)}
                presupuesto = {"dinero": dinero_p, "horas": horas_p} if usar_presupuesto else None
                model = MODEL_REGISTRY[engine]({**S["config"].get("abm", {}), "beta": beta})
                df_final, tray, _, ev = model.run(df, adj, steps=steps, seed=seed,
                                                  evento=evento, presupuesto=presupuesto)
                S.update({"df_final": df_final, "tray": tray,
                          "events": ev[:CONFIG["eventos_max_guardados"]],
                          "n_events_total": len(ev)})
                st.success(f"{engine} terminado | {len(ev)} eventos de influencia registrados "
                           f"(guardados: {min(len(ev), CONFIG['eventos_max_guardados'])})")
        tray = S.get("tray", pd.DataFrame())
        if not tray.empty:
            st.line_chart(tray.set_index("step")[["SIMPATIZANTE", "OPOSITOR", "INDECISO"]])
            if "dinero_restante" in tray.columns:
                st.line_chart(tray.set_index("step")[["dinero_restante", "horas_restante"]])

# ---------- TAB 3 ----------
with tabs[2]:
    if df.empty:
        st.info("➡️ Genera el universo primero")
    else:
        st.subheader("Visualización ABM + Campos de Acción Estratégica")
        sub = st.tabs(["🗺️ Mapa real (SAF)", "🎮 Espacio NetLogo-style", "🔥 Calor",
                       "🚚 Rutas", "🌊 Ondas", "🕸️ Red de centros"])
        df_viz = df.copy()
        df_saf_tmp = Indicadores.saf(df)
        df_viz["campo"] = df_viz["territorial_unit_id"].map(dict(zip(df_saf_tmp["territorio"], df_saf_tmp["campo"])))
        df_viz["color_rgba"] = df_viz["intencion"].map(COLOR_INTENCION)
        df_viz["radius"] = (df_viz["influencia_SAF"] * 180 + 40).astype(int)

        with sub[0]:
            color_mode = st.radio("Colorear por", ["intencion", "campo"], horizontal=True, key="map_color")
            show_links = st.checkbox("Mostrar lazos de red (top influencia)", value=False, key="map_links")
            top_max = max(50, len(df))
            top_show = st.slider("Top N agentes (legibilidad)", 50, top_max, min(200, top_max), key="map_top")
            df_map = df_viz.nlargest(top_show, "influencia_SAF").copy()
            df_map["color_rgba"] = (df_map["campo"].map(CAMPO_RGBA) if color_mode == "campo"
                                    else df_map["intencion"].map(COLOR_INTENCION))
            if HAS_PYDECK:
                estilo = st.selectbox("Estilo de mapa", ["claro (sin token)", "oscuro (sin token)",
                                                         "calles (requiere MAPBOX_TOKEN)"], key="map_style")
                map_style = {"claro (sin token)": pdk.map_styles.LIGHT,
                             "oscuro (sin token)": pdk.map_styles.DARK,
                             "calles (requiere MAPBOX_TOKEN)": "mapbox://styles/mapbox/streets-v12"}[estilo]
                layers = [pdk.Layer("ScatterplotLayer", data=df_map, get_position="[lon, lat]",
                                    get_fill_color="color_rgba", get_radius="radius", pickable=True,
                                    opacity=0.85, stroked=True, get_line_color=[255, 255, 255, 80],
                                    line_width_min_pixels=1)]
                if "es_broker_insertado" in df_map.columns:
                    brokers = df_map[df_map["es_broker_insertado"]]
                    if not brokers.empty:
                        layers.append(pdk.Layer("ScatterplotLayer", data=brokers, get_position="[lon, lat]",
                                                get_fill_color=[255, 215, 0, 255], get_radius=280, pickable=True))
                if show_links and len(df_map) > 5:
                    edges, coords = [], df_map[["lon", "lat"]].values
                    for i in range(min(40, len(coords))):
                        for j in range(i + 1, min(40, len(coords))):
                            if ((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5 < 0.06:
                                edges.append({"source": [coords[i][0], coords[i][1]],
                                              "target": [coords[j][0], coords[j][1]]})
                    if edges:
                        layers.append(pdk.Layer("LineLayer", data=edges, get_source_position="source",
                                                get_target_position="target", get_color=[180, 180, 180, 90], get_width=2))
                view = pdk.ViewState(latitude=19.40, longitude=-99.14, zoom=9.2, pitch=35, bearing=0)
                tooltip = {"html": "<b>{agent_id}</b><br/>Territorio: {territorial_unit_id}<br/>"
                                   "Intención: {intencion}<br/>Campo: {campo}<br/>"
                                   "Influencia: {influencia_SAF}<br/>Habilidades: {habilidades_sociales}",
                           "style": {"backgroundColor": "#1a1a2e", "color": "white"}}
                st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view,
                                         map_style=map_style, tooltip=tooltip))
            else:
                st.map(df_map)
                st.warning("Instala pydeck: `pip install pydeck`")
            c1, c2, c3 = st.columns(3)
            c1.markdown("🟢 **CONSOLIDACION** — simpat ≥ 50%")
            c2.markdown("🔴 **DISPUTA_ABIERTA** — indecisos ≥ 35%")
            c3.markdown("🔵 **CONTENCION** — resto")
            st.dataframe(df_saf_tmp[["territorio", "campo", "dominante", "dominancia",
                                     "estabilidad", "institucionalizacion"]], use_container_width=True)

        with sub[1]:
            metrica_size = st.selectbox("Tamaño por", ["influencia_SAF", "habilidades_sociales",
                                                       "grado", "temperatura_sintetica"], key="size_nl")
            color_by = st.selectbox("Color por", ["intencion", "campo"], key="color_nl")
            show_only_top = st.checkbox("Solo top 30% influencia", value=False, key="nl_top")
            nl_links = st.checkbox("Lazos entre agentes cercanos", value=True, key="nl_links")
            df_nl = df_viz.copy()
            if show_only_top:
                df_nl = df_nl[df_nl["influencia_SAF"] >= df_nl["influencia_SAF"].quantile(0.70)]
            color_col = "intencion" if color_by == "intencion" else "campo"
            color_map = COLOR_HEX if color_by == "intencion" else CAMPO_COLOR
            if HAS_PLOTLY:
                fig = px.scatter(df_nl, x="lon", y="lat", color=color_col, size=metrica_size,
                                 hover_name="agent_id", color_discrete_map=color_map,
                                 hover_data=["territorial_unit_id", "habilidades_sociales", "grado",
                                             "influencia_SAF", "intencion", "campo"],
                                 title="Espacio de agentes (NetLogo-style) — roles SAF", size_max=32)
                fig.update_layout(height=650, template="plotly_dark")
                if "es_broker_insertado" in df_nl.columns:
                    brokers = df_nl[df_nl["es_broker_insertado"]]
                    if not brokers.empty:
                        fig.add_trace(go.Scatter(x=brokers["lon"], y=brokers["lat"], mode="markers",
                                                 marker=dict(symbol="star", size=20, color="gold",
                                                             line=dict(width=1, color="white")),
                                                 name="Broker SAF", hovertext=brokers["agent_id"]))
                if nl_links:
                    sample = df_nl.nlargest(25, "influencia_SAF")
                    for i, r1 in sample.iterrows():
                        for j, r2 in sample.iterrows():
                            if i >= j:
                                continue
                            if ((r1["lon"] - r2["lon"]) ** 2 + (r1["lat"] - r2["lat"]) ** 2) ** 0.5 < 0.05:
                                fig.add_trace(go.Scatter(x=[r1["lon"], r2["lon"]], y=[r1["lat"], r2["lat"]],
                                                         mode="lines", line=dict(color="rgba(180,180,180,0.35)", width=1),
                                                         showlegend=False, hoverinfo="skip"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.map(df_nl)
            st.write("**Top 12 actores por influencia_SAF** (centros del campo)")
            st.dataframe(df_viz.nlargest(12, "influencia_SAF")[
                ["agent_id", "territorial_unit_id", "intencion", "campo",
                 "habilidades_sociales", "grado", "influencia_SAF"]], use_container_width=True)

        with sub[2]:
            metrica = st.selectbox("Métrica de calor", ["influencia_SAF", "polarizacion_local",
                                                        "temperatura_sintetica", "habilidades_sociales"], key="calor")
            if HAS_PYDECK:
                st.pydeck_chart(pdk.Deck(
                    layers=[pdk.Layer("HeatmapLayer", data=df, get_position="[lon, lat]",
                                      get_weight=metrica, radius_pixels=60)],
                    initial_view_state=pdk.ViewState(latitude=19.40, longitude=-99.15, zoom=9, pitch=0),
                    map_style=pdk.map_styles.DARK))
            else:
                st.map(df)

        with sub[3]:
            n_brig = st.slider("N brigadas", 1, 8, 4, key="nb")
            pasos_b = st.slider("Pasos", 3, 15, 8, key="pb")
            if st.button("Generar rutas"):
                S["rutas"] = VisualMaps.generar_rutas(n_brig, pasos_b, seed=seed)
            rutas = S.get("rutas", pd.DataFrame())
            if not rutas.empty:
                if HAS_PYDECK:
                    path_data = [{"brigada": b, "path": sub[["lon", "lat"]].values.tolist()}
                                 for b, sub in rutas.groupby("brigada")]
                    st.pydeck_chart(pdk.Deck(
                        layers=[pdk.Layer("PathLayer", data=path_data, get_path="path",
                                          get_color=[200, 30, 0, 180], width_scale=8, width_min_pixels=3),
                                pdk.Layer("ScatterplotLayer", data=rutas, get_position="[lon, lat]",
                                          get_fill_color=[200, 30, 0, 200], get_radius=80)],
                        initial_view_state=pdk.ViewState(latitude=19.40, longitude=-99.14, zoom=9),
                        map_style=pdk.map_styles.LIGHT))
                else:
                    st.map(rutas)

        with sub[4]:
            origen = st.selectbox("Origen de la onda", TERRITORIOS, key="origen_onda")
            pasos_o = st.slider("Pasos de onda", 3, 15, 10, key="po")
            if st.button("Generar ondas de difusión"):
                S["ondas"] = VisualMaps.generar_ondas(origen, pasos_o, seed=seed)
            ondas = S.get("ondas", pd.DataFrame())
            if not ondas.empty and HAS_PYDECK:
                st.pydeck_chart(pdk.Deck(
                    layers=[pdk.Layer("ScatterplotLayer", data=ondas, get_position="[lon, lat]",
                                      get_fill_color="color_rgba", get_radius=900, pickable=True)],
                    initial_view_state=pdk.ViewState(latitude=19.40, longitude=-99.14, zoom=8.5),
                    map_style=pdk.map_styles.DARK,
                    tooltip={"html": "<b>step {step}</b> radio {radio_km} km · intensidad {intensidad:.0f}"}))

        with sub[5]:
            if HAS_PLOTLY:
                fig = go.Figure()
                for s, t, w in EDGES_TERR:
                    x0, y0 = TERRITORIOS_COORDS[s][1], TERRITORIOS_COORDS[s][0]
                    x1, y1 = TERRITORIOS_COORDS[t][1], TERRITORIOS_COORDS[t][0]
                    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                             line=dict(color="rgba(150,150,150,0.5)", width=2 * w),
                                             showlegend=False, hoverinfo="skip"))
                sizes = df.groupby("territorial_unit_id").size().reindex(TERRITORIOS).fillna(1)
                campos = dict(zip(df_saf_tmp["territorio"], df_saf_tmp["campo"]))
                fig.add_trace(go.Scatter(
                    x=[TERRITORIOS_COORDS[t][1] for t in TERRITORIOS],
                    y=[TERRITORIOS_COORDS[t][0] for t in TERRITORIOS],
                    mode="markers+text", text=TERRITORIOS, textposition="top center",
                    marker=dict(size=np.sqrt(sizes.values) * 4,
                                color=[CAMPO_COLOR.get(campos.get(t), "#95a5a6") for t in TERRITORIOS]),
                    hovertemplate="%{text}<br>n=%{marker.size}<extra></extra>"))
                fig.update_layout(title="Red de centros territoriales (tamaño=n agentes, color=campo SAF)",
                                  height=520, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 4 ----------
with tabs[3]:
    if df.empty:
        st.info("➡️ Genera el universo primero")
    else:
        st.subheader("Monte Carlo — distribución de resultados")
        mc_engine = st.selectbox("Motor", ["abm_saf", "voter_saf", "deffuant_saf"], key="mc_engine")
        if st.button("🎲 Correr Monte Carlo", type="primary"):
            bar = st.progress(0.0, text="Simulando...")
            S["mc"] = run_montecarlo(df, adj, engine=mc_engine, steps=steps, n_sim=n_sim,
                                     seed=seed, beta=beta, params=S["config"].get("abm", {}),
                                     progress=lambda p: bar.progress(min(p, 1.0), text=f"{int(p*100)}%"))
            bar.empty()
            st.success(f"{n_sim} corridas con {mc_engine}")
        mc = S.get("mc")
        if mc:
            sm = mc["summary"]
            if HAS_PLOTLY:
                fig = go.Figure()
                colors = {"SIMPATIZANTE": "#2ecc71", "OPOSITOR": "#e74c3c", "INDECISO": "#95a5a6"}
                for col, c in colors.items():
                    fig.add_trace(go.Scatter(x=sm["step"], y=sm[f"{col}_hi"], mode="lines",
                                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=sm["step"], y=sm[f"{col}_lo"], mode="lines",
                                             fill="tonexty", fillcolor=c.replace("#", "rgba(").replace(")", "") + ",0.15)"
                                             if False else f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.15)",
                                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=sm["step"], y=sm[f"{col}_mean"], mode="lines",
                                             name=col, line=dict(color=c, width=2)))
                st.plotly_chart(fig, use_container_width=True)
                st.plotly_chart(px.histogram(mc["finales"], x="SIMPATIZANTE", nbins=25,
                                             title="Distribución final de simpatía (todas las corridas)"),
                                use_container_width=True)
            st.markdown("**Probabilidad de estado de campo por territorio (a través de corridas)**")
            st.dataframe(mc["campos_prob"].pivot_table(index="territorio", columns="campo",
                                                       values="prob", fill_value=0).round(3),
                         use_container_width=True)
            fin = mc["finales"]["SIMPATIZANTE"]
            st.metric("Simpatía final (media ± sd)", f"{fin.mean():.3f} ± {fin.std():.3f}",
                      f"p5={fin.quantile(0.05):.3f} · p95={fin.quantile(0.95):.3f}")

# ---------- TAB 5 ----------
with tabs[4]:
    if df.empty:
        st.info("➡️ Genera el universo primero")
    else:
        st.subheader("Broker vs Adversario — simulación de contención")
        cA, cB = st.columns(2)
        with cA:
            if st.button("🧠 Insertar broker (parámetros del sidebar)"):
                df2, adj2, bidx = Lab.insertar_broker(
                    df, {k: list(v) for k, v in adj.items()}, territorio_broker, lat_b, lon_b,
                    capital, acceso, lider, arraigo, mov, desconf, intencion_b,
                    grado_objetivo=grado_b, seed=seed, cfg=S["config"])
                S["df"], S["adj"] = df2, adj2
                S["broker_ids"].append(df2.loc[bidx, "agent_id"])
                st.success(f"Broker insertado: {df2.loc[bidx, 'agent_id']} en {territorio_broker}")
        with cB:
            adv_terr = st.selectbox("Territorio del adversario", TERRITORIOS, key="adv_terr")
            adv_int = st.selectbox("Intención del adversario",
                                   [x for x in ["OPOSITOR", "SIMPATIZANTE", "INDECISO"] if x != intencion_b], key="adv_int")
            if st.button("⚔️ Insertar adversario"):
                df2, adj2, aidx = Lab.insertar_broker(
                    df, {k: list(v) for k, v in adj.items()}, adv_terr,
                    TERRITORIOS_COORDS[adv_terr][0], TERRITORIOS_COORDS[adv_terr][1],
                    capital, acceso, lider, arraigo, mov, desconf, adv_int,
                    grado_objetivo=grado_b, seed=seed + 7, cfg=S["config"], es_adversario=True)
                S["df"], S["adj"] = df2, adj2
                S["adv_ids"].append(df2.loc[aidx, "agent_id"])
                st.success(f"Adversario insertado: {df2.loc[aidx, 'agent_id']} en {adv_terr}")
        if S["broker_ids"] or S["adv_ids"]:
            st.caption(f"Brokers: {', '.join(S['broker_ids']) or '—'} | "
                       f"Adversarios: {', '.join(S['adv_ids']) or '—'}")
            if st.button("▶️ Correr ABM con duelos"):
                model = ABMSAF({**S["config"].get("abm", {}), "beta": beta})
                df_fin, tray, conv, ev = model.run(S["df"], S["adj"], steps=steps, seed=seed)
                S.update({"df_final": df_fin, "tray": tray, "events": ev[:CONFIG["eventos_max_guardados"]],
                          "n_events_total": len(ev)})
                st.line_chart(tray.set_index("step")[["SIMPATIZANTE", "OPOSITOR", "INDECISO"]])
                duel = df_fin[df_fin["es_broker_insertado"]][["agent_id", "intencion", "conversiones_causadas"]]
                st.markdown("**Duelo de conversión (evento → evento, evidencia SIM)**")
                st.dataframe(duel, use_container_width=True)
        st.markdown("---")
        st.subheader("🎯 SPOF — puntos únicos de falla del campo")
        if st.button("Analizar SPOF"):
            with st.spinner("Analizando fragmentación..."):
                spof_df = spof_analisis(df, adj, top_k=S["config"]["analisis"]["spof_top_k"])
            st.dataframe(spof_df, use_container_width=True)
            if not spof_df.empty and spof_df.iloc[0]["delta_fragmentacion"] > 0.1:
                st.warning(f"⚠️ {spof_df.iloc[0]['agent_id']} es crítico: su remoción fragmenta la red "
                           f"(Δ={spof_df.iloc[0]['delta_fragmentacion']}).")
            else:
                st.info("Red robusta: ningún actor individual concentra la cohesión.")
        st.markdown("---")
        st.subheader("🧪 Optimizador de inserción de broker")
        st.caption("Búsqueda en grilla (territorio × capital × grado) sobre un subuniverso, "
                   "score = share final de la intención objetivo.")
        if st.button("Optimizar"):
            bar = st.progress(0.0, text="Optimizando...")
            res_opt, base = optimizar_broker(df, adj, intencion=intencion_b, cfg=S["config"],
                                             beta=beta, progress=lambda p: bar.progress(min(p, 1.0)))
            bar.empty()
            st.metric("Línea base (sin broker)", f"{base:.4f}")
            st.dataframe(res_opt.head(12), use_container_width=True)
            best = res_opt.iloc[0]
            st.success(f"Mejor combinación: {best['territorio']} · capital={best['capital']} · "
                       f"grado={int(best['grado'])} · score={best['score']} (Δ={best['delta_vs_base']:+.4f})")
            if st.button(f"Aplicar mejor: broker en {best['territorio']}"):
                df2, adj2, bidx = Lab.insertar_broker(
                    df, {k: list(v) for k, v in adj.items()}, str(best["territorio"]),
                    TERRITORIOS_COORDS[str(best["territorio"])][0], TERRITORIOS_COORDS[str(best["territorio"])][1],
                    float(best["capital"]), float(best["capital"]), float(best["capital"]),
                    0.7, 0.75, 0.15, intencion_b, grado_objetivo=int(best["grado"]),
                    seed=seed, cfg=S["config"])
                S["df"], S["adj"] = df2, adj2
                S["broker_ids"].append(df2.loc[bidx, "agent_id"])
                st.rerun()

# ---------- TAB 6 ----------
with tabs[5]:
    st.subheader("Influencia dirigida — ¿desde dónde se dirige la influencia?")
    st.caption("Regla de oro: nunca «X influye en Y» sin Fuente + Receptor + Mecanismo + Campo + "
               "Tiempo + Evidencia + Confianza. Escala: E0–E5 (empírica) / SIM (motor simulado).")
    with st.expander("📚 Escala de evidencia"):
        st.json(EVIDENCIA_DESC)
    if df.empty:
        st.info("➡️ Genera el universo y corre el ABM para poblar eventos de influencia.")
    else:
        graph = FieldGraph(df, adj, S.get("events", []))
        top_ids = df.nlargest(30, "influencia_SAF")["agent_id"].tolist()
        all_ids = top_ids + S["broker_ids"] + S["adv_ids"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Nivel 1–2: ¿Quién influye sobre X?**")
            tgt = st.selectbox("Actor objetivo", all_ids, key="tgt_inf")
            fld = st.selectbox("Filtrar por campo/territorio", ["(todos)"] + TERRITORIOS, key="fld_inf")
            if st.button("Consultar influencia_sources"):
                df_ev, df_pot = graph.influence_sources(tgt, None if fld == "(todos)" else fld)
                st.markdown(f"Influencia **observada** ({len(df_ev)} eventos):")
                st.dataframe(df_ev, use_container_width=True)
                st.markdown(f"Influencia **potencial** por estructura de red (E2, no causal):")
                st.dataframe(df_pot, use_container_width=True)
        with c2:
            st.markdown("**Nivel 3: rutas de influencia (actor-actor y campo-mediadas)**")
            src = st.selectbox("Actor fuente", all_ids, key="src_inf")
            tgt2 = st.selectbox("Actor destino", [a for a in all_ids if a != src], key="tgt_path")
            hops = st.number_input("Máx. saltos", 1, 5, 3, key="hops")
            if st.button("Consultar influence_paths"):
                paths = graph.influence_paths(src, tgt2, max_hops=int(hops))
                if paths:
                    for pth in paths:
                        st.markdown(f"`{' → '.join(pth['path'])}` — {pth['hops']} saltos · {pth['tipo']}")
                else:
                    st.info("Sin rutas dentro del límite de saltos.")
        st.markdown("---")
        st.markdown("**Matriz de influencia (agregada)**")
        matriz = graph.matriz()
        if matriz is not None and not matriz.empty:
            st.dataframe(matriz, use_container_width=True)
        else:
            st.info("Sin eventos: corre el ABM (tab 2) primero. Los eventos SIM nunca se presentan "
                    "como causalidad empírica.")

# ---------- TAB 7 ----------
with tabs[6]:
    st.subheader("📋 77 Preguntas de investigación — respuestas calculadas")
    st.caption("Toda respuesta se deriva de los indicadores actuales. Evidencia SIM-CALC: "
               "universo sintético/calibrado, no observación empírica.")
    if df.empty:
        st.info("➡️ Genera el universo primero.")
    else:
        preguntas = build_preguntas(df, S.get("df_terr", Indicadores.territoriales(df)),
                                    S.get("df_saf", Indicadores.saf(df)),
                                    S.get("red_ind") or Indicadores.red(df, adj),
                                    S.get("inf_ind") or Indicadores.influencia(df))
        cat = st.selectbox("Filtrar categoría", ["(todas)"] + sorted(preguntas["categoria"].unique().tolist()))
        st.dataframe(preguntas if cat == "(todas)" else preguntas[preguntas["categoria"] == cat],
                     use_container_width=True)
        st.caption(f"Total: {len(preguntas)} preguntas.")

# ---------- TAB 8 ----------
with tabs[7]:
    st.subheader("📤 Experimento — run() · validate() · save()")
    st.caption("Patrón de reproducibilidad: mismo dataset, mismo seed, mismo config → mismo hash.")
    if df.empty:
        st.info("➡️ Genera el universo y corre una simulación primero.")
    else:
        exp_name = st.text_input("Nombre del experimento", "xochimilco_field_001")
        exp_engine = st.selectbox("Motor", ["abm_saf", "voter_saf", "deffuant_saf"], key="exp_engine")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("1️⃣ run()", type="primary"):
                exp = Experiment(exp_name, exp_engine, S["config"], seed=int(seed))
                exp.run(df, adj, steps=steps, beta=beta)
                S["experimento"] = exp
                st.success("Ejecutado.")
        with col2:
            if st.button("2️⃣ validate()"):
                exp = S.get("experimento")
                if exp is None:
                    st.warning("Corre run() primero.")
                else:
                    checks = exp.validate()
                    for c in checks:
                        (st.success if c["ok"] else st.error)(f"**{c['check']}** — {c['detalle']}")
        with col3:
            if st.button("3️⃣ save() + descarga"):
                exp = S.get("experimento")
                if exp is None:
                    st.warning("Corre run() primero.")
                else:
                    payload = exp.payload(S.get("df_terr", Indicadores.territoriales(df)),
                                          S.get("df_saf", Indicadores.saf(df)),
                                          S.get("red_ind") or Indicadores.red(df, adj),
                                          S.get("inf_ind") or Indicadores.influencia(df),
                                          extra={"montecarlo": (S["mc"]["summary"].to_dict(orient="records")
                                                                if S.get("mc") else None),
                                                 "calibracion": {"source": S.get("calib_source"),
                                                                 "hash": S.get("calib_hash")},
                                                 "n_events_total": S.get("n_events_total", 0)})
                    path = exp.save(payload)
                    st.success(f"Guardado: `{path}` · hash=`{payload['metadata']['output_hash'][:16]}…`")
                    st.download_button("⬇️ Descargar JSON", data=json.dumps(payload, ensure_ascii=False,
                                                                           default=str, indent=2),
                                       file_name=f"{exp_name}.json", mime="application/json")

# ---------- TAB 9 ----------
with tabs[8]:
    st.subheader("📥 Calibración con datos públicos (agregados, sin PII)")
    st.markdown("Formato combinado (recomendado): "
                "`territorio,simpat_pct,opos_pct,indec_pct,conflicto_pct,inseguridad_pct,"
                "desconfianza_proxy,movilizacion_proxy`")
    with st.expander("📋 Plantilla CSV"):
        st.code(Calibrador.TEMPLATE_COMBINADO, language="csv")
    up = st.file_uploader("Sube CSV de calibración (ENSU/INE-like agregado)", type=["csv"])
    txt = st.text_area("…o pega el CSV aquí", height=140, key="calib_txt")
    if st.button("Cargar calibración"):
        try:
            src_cal = up if up is not None else (txt if txt.strip() else Calibrador.TEMPLATE_COMBINADO)
            raw = src_cal.read() if hasattr(src_cal, "read") else str(src_cal)
            df_cal = Calibrador.parse_csv(StringIO(raw.decode() if isinstance(raw, bytes) else raw))
            params = Calibrador.build_params(df_cal)
            if not params:
                st.error("Ningún territorio del CSV coincide con los territorios del sistema.")
            else:
                S.update({"calib_params": params,
                          "calib_source": up.name if up is not None else "texto_manual",
                          "calib_hash": sha256(raw if isinstance(raw, str) else raw.decode())})
                st.success(f"Calibración cargada: {len(params)} territorios · "
                           f"hash={S['calib_hash'][:12]}…")
                st.dataframe(pd.DataFrame(params).T, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    if S.get("calib_params"):
        st.info(f"Calibración activa: {S['calib_source']} (hash {S['calib_hash'][:12]}…). "
                "Activa «Usar calibración» en el sidebar y regenera el universo.")
