# =============================================================================
# MONOLITO SITER-CAE v5.0 - UNIFICADO
# =============================================================================

import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIGURACIÓN Y CONSTANTES GLOBALES
# -----------------------------------------------------------------------------
CONFIG = {
    "habilidades": {
        "capital_social": 0.25,
        "acceso_informacion": 0.20,
        "influencia_liderazgo": 0.20,
        "arraigo": 0.15,
        "nivel_movilizacion": 0.10,
        "desconfianza": 0.10,
    }
}

ALCALDIAS_CDMX = [
    "ALVARO OBREGON",
    "AZCAPOTZALCO",
    "BENITO JUAREZ",
    "COYOACAN",
    "CUAJIMALPA DE MORELOS",
    "CUAUHTEMOC",
    "GUSTAVO A MADERO",
    "IZTACALCO",
    "IZTAPALAPA",
    "LA MAGDALENA CONTRERAS",
    "MIGUEL HIDALGO",
    "MILPA ALTA",
    "TLAHUAC",
    "TLALPAN",
    "VENUSTIANO CARRANZA",
    "XOCHIMILCO",
]

ALCALDIA_COORDS = {
    "CUAUHTEMOC": (19.4326, -99.1332),
    "BENITO JUAREZ": (19.3984, -99.1576),
    "MIGUEL HIDALGO": (19.4285, -99.2000),
    "COYOACAN": (19.3467, -99.1617),
    "IZTAPALAPA": (19.3550, -99.0620),
    "GUSTAVO A MADERO": (19.4900, -99.1100),
    "ALVARO OBREGON": (19.3580, -99.2270),
    "TLALPAN": (19.2880, -99.1670),
    "XOCHIMILCO": (19.2630, -99.1040),
    "VENUSTIANO CARRANZA": (19.4200, -99.1000),
    "AZCAPOTZALCO": (19.4870, -99.1860),
    "IZTACALCO": (19.3950, -99.0980),
    "CUAJIMALPA DE MORELOS": (19.3570, -99.2900),
    "LA MAGDALENA CONTRERAS": (19.3200, -99.2400),
    "TLAHUAC": (19.2700, -99.0050),
    "MILPA ALTA": (19.1920, -99.0230),
}

CONTIGUIDAD_ALCALDIAS = {
    "CUAUHTEMOC": [
        "MIGUEL HIDALGO",
        "BENITO JUAREZ",
        "VENUSTIANO CARRANZA",
        "AZCAPOTZALCO",
        "GUSTAVO A MADERO",
        "IZTACALCO",
    ],
    "BENITO JUAREZ": [
        "CUAUHTEMOC",
        "COYOACAN",
        "IZTACALCO",
        "MIGUEL HIDALGO",
        "ALVARO OBREGON",
    ],
    "IZTAPALAPA": [
        "IZTACALCO",
        "TLAHUAC",
        "XOCHIMILCO",
        "COYOACAN",
        "VENUSTIANO CARRANZA",
    ],
    "GUSTAVO A MADERO": ["AZCAPOTZALCO", "VENUSTIANO CARRANZA", "CUAUHTEMOC"],
    "COYOACAN": [
        "BENITO JUAREZ",
        "TLALPAN",
        "XOCHIMILCO",
        "IZTAPALAPA",
        "ALVARO OBREGON",
    ],
    "TLALPAN": [
        "COYOACAN",
        "XOCHIMILCO",
        "LA MAGDALENA CONTRERAS",
        "ALVARO OBREGON",
    ],
    "XOCHIMILCO": ["TLALPAN", "COYOACAN", "IZTAPALAPA", "TLAHUAC", "MILPA ALTA"],
    "ALVARO OBREGON": [
        "COYOACAN",
        "TLALPAN",
        "LA MAGDALENA CONTRERAS",
        "CUAJIMALPA DE MORELOS",
        "MIGUEL HIDALGO",
        "BENITO JUAREZ",
    ],
    "MIGUEL HIDALGO": [
        "CUAUHTEMOC",
        "AZCAPOTZALCO",
        "BENITO JUAREZ",
        "ALVARO OBREGON",
        "CUAJIMALPA DE MORELOS",
    ],
    "AZCAPOTZALCO": ["GUSTAVO A MADERO", "CUAUHTEMOC", "MIGUEL HIDALGO"],
    "IZTACALCO": ["BENITO JUAREZ", "CUAUHTEMOC", "VENUSTIANO CARRANZA", "IZTAPALAPA"],
    "VENUSTIANO CARRANZA": [
        "GUSTAVO A MADERO",
        "CUAUHTEMOC",
        "IZTACALCO",
        "IZTAPALAPA",
    ],
    "CUAJIMALPA DE MORELOS": [
        "ALVARO OBREGON",
        "MIGUEL HIDALGO",
        "LA MAGDALENA CONTRERAS",
    ],
    "LA MAGDALENA CONTRERAS": ["ALVARO OBREGON", "TLALPAN", "CUAJIMALPA DE MORELOS"],
    "TLAHUAC": ["IZTAPALAPA", "XOCHIMILCO", "MILPA ALTA"],
    "MILPA ALTA": ["XOCHIMILCO", "TLAHUAC", "TLALPAN"],
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# SECCIÓN 1a · CORE Y EXPERIMENT
# =============================================================================


class Experiment:
    """Contenedor de ejecución y metadatos de un experimento."""

    def __init__(
        self,
        df: pd.DataFrame,
        adj: dict,
        data_meta: dict,
        beta: float = 1.2,
        steps: int = 15,
    ):
        self.df_init = df.copy()
        self.adj = adj
        self.data_meta = data_meta
        self.beta = beta
        self.steps = steps

    def to_payload(
        self, df_final: pd.DataFrame, trajectory: pd.DataFrame
    ) -> Dict[str, Any]:
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_origin": self.data_meta,
            "params": {"beta": self.beta, "steps": self.steps},
            "initial_state": {
                "n_nodes": len(self.df_init),
                "simpatizantes": int((self.df_init["spin"] == 1).sum()),
                "opositores": int((self.df_init["spin"] == -1).sum()),
                "indecisos": int((self.df_init["spin"] == 0).sum()),
            },
            "final_state": {
                "simpatizantes": int((df_final["spin"] == 1).sum()),
                "opositores": int((df_final["spin"] == -1).sum()),
                "indecisos": int((df_final["spin"] == 0).sum()),
            },
            "trajectory_summary": trajectory.to_dict(orient="records"),
        }


# =============================================================================
# SECCIÓN 1b · LAB (GENERADOR SINTÉTICO BASE)
# =============================================================================


class Lab:
    @staticmethod
    def generate(
        n: int = 400,
        seed: int = 42,
        p_intra: float = 0.05,
        p_inter: float = 0.01,
        cfg: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, dict]:
        cfg = cfg or CONFIG
        rng = np.random.default_rng(seed)

        rows = []
        for i in range(n):
            alc = str(rng.choice(ALCALDIAS_CDMX))
            lat, lon = ALCALDIA_COORDS[alc]
            opinion = float(rng.uniform(-0.8, 0.8))

            cap_soc = float(rng.uniform(0.1, 0.9))
            acc_inf = float(rng.uniform(0.1, 0.9))
            inf_lid = float(rng.uniform(0.1, 0.9))
            arr = float(rng.uniform(0.1, 0.9))
            niv_mov = float(rng.uniform(0.1, 0.9))
            desconf = float(rng.uniform(0.1, 0.9))

            rows.append(
                {
                    "agent_id": f"AG-{i:04d}",
                    "territorial_unit_id": alc,
                    "seccion": f"S{i:04d}",
                    "alcaldia": alc,
                    "lat": lat + float(rng.normal(0, 0.01)),
                    "lon": lon + float(rng.normal(0, 0.01)),
                    "opinion_continua": opinion,
                    "capital_social": cap_soc,
                    "acceso_informacion": acc_inf,
                    "influencia_liderazgo": inf_lid,
                    "arraigo": arr,
                    "nivel_movilizacion": niv_mov,
                    "desconfianza": desconf,
                    "exposicion_problema": float(rng.uniform(0.1, 0.9)),
                    "resistencia_institucional": float(rng.uniform(0.1, 0.9)),
                    "temperatura_sintetica": float(rng.uniform(20.0, 80.0)),
                    "fatiga": 0.0,
                    "capital_politico": float(rng.uniform(0.3, 0.9)),
                    "es_broker_insertado": False,
                    "es_adversario": False,
                    "polarizacion_local": float(rng.uniform(0.1, 0.7)),
                    "calibrado": False,
                }
            )

        df = pd.DataFrame(rows)
        df["spin"] = np.where(
            df["opinion_continua"] > 0.25,
            1,
            np.where(df["opinion_continua"] < -0.25, -1, 0),
        )
        df["intencion"] = np.where(
            df["spin"] == 1,
            "SIMPATIZANTE",
            np.where(df["spin"] == -1, "OPOSITOR", "INDECISO"),
        )

        w = cfg["habilidades"]
        df["habilidades_sociales"] = (
            df["capital_social"] * w["capital_social"]
            + df["acceso_informacion"] * w["acceso_informacion"]
            + df["influencia_liderazgo"] * w["influencia_liderazgo"]
            + df["arraigo"] * w["arraigo"]
            + df["nivel_movilizacion"] * w["nivel_movilizacion"]
            + df["desconfianza"] * w["desconfianza"]
        ).clip(0, 1)

        # Adyacencia por bloques de alcaldía
        adj = defaultdict(list)
        for alc in df["alcaldia"].unique():
            idxs = df.index[df["alcaldia"] == alc].tolist()
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    if float(rng.random()) < p_intra:
                        x, y = idxs[a], idxs[b]
                        adj[x].append(y)
                        adj[y].append(x)

        for a1, vecinos in CONTIGUIDAD_ALCALDIAS.items():
            idxs1 = df.index[df["alcaldia"] == a1].tolist()
            for a2 in vecinos:
                idxs2 = df.index[df["alcaldia"] == a2].tolist()
                for x in idxs1:
                    for y in idxs2:
                        if float(rng.random()) < p_inter:
                            adj[x].append(y)
                            adj[y].append(x)

        df["grado"] = df.index.map(lambda i: len(adj.get(i, [])))
        max_g = max(int(df["grado"].max()), 1)
        df["influencia_SAF"] = (
            df["habilidades_sociales"] * 0.6 + (df["grado"] / max_g) * 0.4
        ).clip(0, 1)

        return df, dict(adj)


# =============================================================================
# SECCIÓN 1c · DATA PROVIDER CDMX (5 modos)
# =============================================================================


class DataProvider:
    """
    Única puerta de entrada de datos.
    Siempre devuelve: (df_state, adj, meta)
    """

    def __init__(
        self, mode: str = "synth_pure", seed: int = 42, cfg: Optional[dict] = None
    ):
        self.mode = mode
        self.seed = seed
        self.cfg = cfg or CONFIG
        self.rng = np.random.default_rng(seed)

    def load(self, **kwargs) -> Tuple[pd.DataFrame, dict, dict]:
        loaders = {
            "real": self._load_real,
            "dummy": self._load_dummy,
            "synth_coherent": self._load_synth_coherent,
            "synth_pure": self._load_synth_pure,
            "synth_calib": self._load_synth_calib,
        }
        if self.mode not in loaders:
            raise ValueError(f"Modo desconocido: {self.mode}")
        return loaders[self.mode](**kwargs)

    # 1. REAL
    def _load_real(
        self,
        secciones_csv=None,
        shp_path=None,
        electoral_csv=None,
        socio_csv=None,
        seguridad_csv=None,
        p_intra: float = 0.08,
        p_inter: float = 0.02,
    ) -> Tuple[pd.DataFrame, dict, dict]:

        if secciones_csv is None:
            raise ValueError("Modo real requiere al menos secciones_csv")

        df = pd.read_csv(secciones_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        df["seccion"] = df["seccion"].astype(str).str.zfill(4)
        df["alcaldia"] = df["alcaldia"].str.upper().str.strip()
        df["territorial_unit_id"] = df["seccion"]
        df["agent_id"] = "SEC-" + df["seccion"]

        if "lat" not in df.columns or "lon" not in df.columns:
            if shp_path is not None:
                try:
                    import geopandas as gpd

                    gdf = gpd.read_file(shp_path)
                    gdf["seccion"] = gdf["seccion"].astype(str).str.zfill(4)
                    centroids = gdf.geometry.centroid
                    gdf["lat"] = centroids.y
                    gdf["lon"] = centroids.x
                    df = df.merge(
                        gdf[["seccion", "lat", "lon"]], on="seccion", how="left"
                    )
                except Exception as e:
                    st.warning(
                        f"No se pudo leer SHP ({e}). Usando centroides de alcaldía."
                    )
                    df["lat"] = df["alcaldia"].map(
                        lambda a: ALCALDIA_COORDS.get(a, (19.43, -99.13))[0]
                    )
                    df["lon"] = df["alcaldia"].map(
                        lambda a: ALCALDIA_COORDS.get(a, (19.43, -99.13))[1]
                    )
            else:
                df["lat"] = df["alcaldia"].map(
                    lambda a: ALCALDIA_COORDS.get(a, (19.43, -99.13))[0]
                )
                df["lon"] = df["alcaldia"].map(
                    lambda a: ALCALDIA_COORDS.get(a, (19.43, -99.13))[1]
                )

        if electoral_csv is not None:
            elec = pd.read_csv(electoral_csv)
            elec.columns = [c.strip().lower() for c in elec.columns]
            elec["seccion"] = elec["seccion"].astype(str).str.zfill(4)

            if "partido_o_coalicion" in elec.columns and "votos" in elec.columns:
                piv = elec.pivot_table(
                    index="seccion",
                    columns="partido_o_coalicion",
                    values="votos",
                    aggfunc="sum",
                    fill_value=0,
                )
                total = piv.sum(axis=1).replace(0, np.nan)
                share = (piv.max(axis=1) / total).fillna(0.33)
                df = df.merge(
                    share.rename("share_max"),
                    left_on="seccion",
                    right_index=True,
                    how="left",
                )
                df["opinion_continua"] = (
                    df["share_max"].fillna(0.33) * 2 - 1
                ).clip(-1, 1)
            else:
                df["opinion_continua"] = 0.0
        else:
            df["opinion_continua"] = 0.0

        df["spin"] = np.where(
            df["opinion_continua"] > 0.25,
            1,
            np.where(df["opinion_continua"] < -0.25, -1, 0),
        )
        df["intencion"] = np.where(
            df["spin"] == 1,
            "SIMPATIZANTE",
            np.where(df["spin"] == -1, "OPOSITOR", "INDECISO"),
        )

        if socio_csv is not None:
            socio = pd.read_csv(socio_csv)
            socio.columns = [c.strip().lower() for c in socio.columns]
            socio["seccion"] = socio["seccion"].astype(str).str.zfill(4)
            df = df.merge(socio, on="seccion", how="left", suffixes=("", "_s"))

        if seguridad_csv is not None:
            seg = pd.read_csv(seguridad_csv)
            seg.columns = [c.strip().lower() for c in seg.columns]
            seg["seccion"] = seg["seccion"].astype(str).str.zfill(4)
            df = df.merge(seg, on="seccion", how="left", suffixes=("", "_q"))

        df = self._fill_required_columns(df)
        adj = self._build_adj_from_alcaldias(df, p_intra=p_intra, p_inter=p_inter)

        df["grado"] = df.index.map(lambda i: len(adj.get(i, [])))
        max_g = max(int(df["grado"].max()), 1)
        df["influencia_SAF"] = (
            df["habilidades_sociales"] * 0.6 + (df["grado"] / max_g) * 0.4
        ).clip(0, 1)

        meta = {
            "mode": "real",
            "source_hash": sha256(str(df[["seccion", "alcaldia"]].values.tolist())),
            "n_nodes": len(df),
            "timestamp": pd.Timestamp.now().isoformat(),
            "notes": "Datos reales CSV (+ SHP si se proporcionó)",
        }
        return df.reset_index(drop=True), dict(adj), meta

    # 2. DUMMY
    def _load_dummy(self, n_secciones: int = 48) -> Tuple[pd.DataFrame, dict, dict]:
        alcaldias = ALCALDIAS_CDMX[:4]
        rows = []
        for i in range(n_secciones):
            alc = alcaldias[i % len(alcaldias)]
            lat, lon = ALCALDIA_COORDS[alc]
            rows.append(
                {
                    "seccion": f"{i+1:04d}",
                    "alcaldia": alc,
                    "lat": lat + float(self.rng.normal(0, 0.01)),
                    "lon": lon + float(self.rng.normal(0, 0.01)),
                    "opinion_continua": float(self.rng.uniform(-0.6, 0.6)),
                    "capital_social": 0.55,
                    "desconfianza": 0.35,
                    "nivel_movilizacion": 0.50,
                }
            )
        df = pd.DataFrame(rows)
        df["territorial_unit_id"] = df["seccion"]
        df["agent_id"] = "DUM-" + df["seccion"]
        df["spin"] = np.where(
            df["opinion_continua"] > 0.25,
            1,
            np.where(df["opinion_continua"] < -0.25, -1, 0),
        )
        df["intencion"] = np.where(
            df["spin"] == 1,
            "SIMPATIZANTE",
            np.where(df["spin"] == -1, "OPOSITOR", "INDECISO"),
        )
        df = self._fill_required_columns(df)
        adj = self._build_adj_from_alcaldias(df, p_intra=0.15, p_inter=0.04)
        df["grado"] = df.index.map(lambda i: len(adj.get(i, [])))
        max_g = max(int(df["grado"].max()), 1)
        df["influencia_SAF"] = (
            df["habilidades_sociales"] * 0.6 + (df["grado"] / max_g) * 0.4
        ).clip(0, 1)

        meta = {
            "mode": "dummy",
            "source_hash": sha256(f"dummy-{n_secciones}-{self.seed}"),
            "n_nodes": len(df),
            "timestamp": pd.Timestamp.now().isoformat(),
            "notes": "Datos dummy mínimos para pruebas rápidas",
        }
        return df.reset_index(drop=True), dict(adj), meta

    # 3. SYNTH_COHERENT
    def _load_synth_coherent(
        self, n: int = 400, stats: Optional[dict] = None
    ) -> Tuple[pd.DataFrame, dict, dict]:
        if stats is None:
            stats = {
                "share_simpat_mean": 0.37,
                "share_simpat_std": 0.14,
                "marginacion_mean": 0.42,
                "marginacion_std": 0.18,
                "participacion_mean": 0.58,
                "participacion_std": 0.11,
                "pob_mean": 2500,
                "pob_std": 900,
            }

        rows = []
        for i in range(n):
            alc = str(self.rng.choice(ALCALDIAS_CDMX))
            lat, lon = ALCALDIA_COORDS[alc]
            share = float(
                np.clip(
                    self.rng.normal(
                        stats["share_simpat_mean"], stats["share_simpat_std"]
                    ),
                    0.05,
                    0.95,
                )
            )
            opinion = 2 * share - 1
            margin = float(
                np.clip(
                    self.rng.normal(
                        stats["marginacion_mean"], stats["marginacion_std"]
                    ),
                    0.05,
                    0.95,
                )
            )
            part = float(
                np.clip(
                    self.rng.normal(
                        stats["participacion_mean"], stats["participacion_std"]
                    ),
                    0.2,
                    0.9,
                )
            )

            rows.append(
                {
                    "seccion": f"S{i:04d}",
                    "alcaldia": alc,
                    "lat": lat + float(self.rng.normal(0, 0.012)),
                    "lon": lon + float(self.rng.normal(0, 0.012)),
                    "opinion_continua": opinion,
                    "capital_social": float(
                        np.clip(1 - margin + self.rng.normal(0, 0.08), 0.1, 0.95)
                    ),
                    "desconfianza": margin,
                    "nivel_movilizacion": part,
                    "pob_aprox": max(
                        400, int(self.rng.normal(stats["pob_mean"], stats["pob_std"]))
                    ),
                }
            )
        df = pd.DataFrame(rows)
        df["territorial_unit_id"] = df["seccion"]
        df["agent_id"] = "COH-" + df["seccion"]
        df["spin"] = np.where(
            df["opinion_continua"] > 0.25,
            1,
            np.where(df["opinion_continua"] < -0.25, -1, 0),
        )
        df["intencion"] = np.where(
            df["spin"] == 1,
            "SIMPATIZANTE",
            np.where(df["spin"] == -1, "OPOSITOR", "INDECISO"),
        )
        df = self._fill_required_columns(df)
        adj = self._build_adj_from_alcaldias(df, p_intra=0.07, p_inter=0.018)
        df["grado"] = df.index.map(lambda i: len(adj.get(i, [])))
        max_g = max(int(df["grado"].max()), 1)
        df["influencia_SAF"] = (
            df["habilidades_sociales"] * 0.6 + (df["grado"] / max_g) * 0.4
        ).clip(0, 1)

        meta = {
            "mode": "synth_coherent",
            "source_hash": sha256(f"coherent-{n}-{self.seed}-{stats}"),
            "n_nodes": len(df),
            "timestamp": pd.Timestamp.now().isoformat(),
            "notes": "Sintético coherente derivado de estadísticas reales",
        }
        return df.reset_index(drop=True), dict(adj), meta

    # 4. SYNTH_PURE
    def _load_synth_pure(
        self, n: int = 400, p_intra: float = 0.05, p_inter: float = 0.01
    ) -> Tuple[pd.DataFrame, dict, dict]:
        df, adj = Lab.generate(
            n=n, seed=self.seed, p_intra=p_intra, p_inter=p_inter, cfg=self.cfg
        )
        df["alcaldia"] = df["territorial_unit_id"]
        meta = {
            "mode": "synth_pure",
            "source_hash": sha256(f"pure-{n}-{self.seed}"),
            "n_nodes": len(df),
            "timestamp": pd.Timestamp.now().isoformat(),
            "notes": "Universo sintético puro (generador original)",
        }
        return df, adj, meta

    # 5. SYNTH_CALIB
    def _load_synth_calib(
        self,
        scenario: str = "polarizacion_alta",
        n: int = 300,
        intensidad: float = 0.8,
    ) -> Tuple[pd.DataFrame, dict, dict]:
        scenarios = {
            "polarizacion_alta": self._scenario_polarizacion_alta,
            "spof_critico": self._scenario_spof_critico,
            "cascade_rapida": self._scenario_cascade_rapida,
            "equilibrio_fragil": self._scenario_equilibrio_fragil,
            "broker_dominante": self._scenario_broker_dominante,
        }
        if scenario not in scenarios:
            raise ValueError(
                f"Escenario desconocido: {scenario}. Opciones: {list(scenarios.keys())}"
            )

        df, adj = scenarios[scenario](n=n, intensidad=intensidad)
        meta = {
            "mode": "synth_calib",
            "scenario": scenario,
            "intensidad": intensidad,
            "source_hash": sha256(f"calib-{scenario}-{n}-{intensidad}-{self.seed}"),
            "n_nodes": len(df),
            "timestamp": pd.Timestamp.now().isoformat(),
            "notes": f"Escenario de calibración: {scenario} (intensidad={intensidad})",
        }
        return df, adj, meta

    # Escenarios específicos
    def _scenario_polarizacion_alta(self, n=300, intensidad=0.8):
        half = n // 2
        rows = []
        for i in range(n):
            bloque = 0 if i < half else 1
            alc = ALCALDIAS_CDMX[bloque * 4 + (i % 4)]
            lat, lon = ALCALDIA_COORDS[alc]
            opinion = (
                (0.7 + 0.25 * intensidad)
                if bloque == 0
                else (-0.7 - 0.25 * intensidad)
            )
            opinion = float(np.clip(opinion + self.rng.normal(0, 0.08), -1, 1))
            rows.append(
                {
                    "seccion": f"P{i:04d}",
                    "alcaldia": alc,
                    "lat": lat + float(self.rng.normal(0, 0.008)),
                    "lon": lon + float(self.rng.normal(0, 0.008)),
                    "opinion_continua": opinion,
                    "capital_social": 0.6,
                    "desconfianza": 0.3 + 0.2 * intensidad,
                    "nivel_movilizacion": 0.65,
                }
            )
        df = pd.DataFrame(rows)
        df = self._finalize_calib_df(df)
        adj = self._build_adj_from_alcaldias(df, p_intra=0.12, p_inter=0.005)
        return self._finish_calib(df, adj)

    def _scenario_spof_critico(self, n=250, intensidad=0.85):
        rows = []
        for i in range(n):
            alc = ALCALDIAS_CDMX[i % 8]
            lat, lon = ALCALDIA_COORDS[alc]
            rows.append(
                {
                    "seccion": f"S{i:04d}",
                    "alcaldia": alc,
                    "lat": lat + float(self.rng.normal(0, 0.01)),
                    "lon": lon + float(self.rng.normal(0, 0.01)),
                    "opinion_continua": float(self.rng.uniform(-0.5, 0.5)),
                    "capital_social": 0.5,
                    "desconfianza": 0.4,
                    "nivel_movilizacion": 0.55,
                }
            )
        df = pd.DataFrame(rows)
        df = self._finalize_calib_df(df)
        adj = self._build_adj_from_alcaldias(df, p_intra=0.06, p_inter=0.01)
        hub = 0
        for j in range(1, min(40, n)):
            if j not in adj[hub]:
                adj[hub].append(j)
                adj[j].append(hub)
        df.loc[hub, "capital_social"] = 0.95
        df.loc[hub, "habilidades_sociales"] = 0.92
        return self._finish_calib(df, adj)

    def _scenario_cascade_rapida(self, n=280, intensidad=0.9):
        rows = []
        for i in range(n):
            alc = ALCALDIAS_CDMX[i % 6]
            lat, lon = ALCALDIA_COORDS[alc]
            opinion = float(self.rng.normal(0.0, 0.15))
            rows.append(
                {
                    "seccion": f"C{i:04d}",
                    "alcaldia": alc,
                    "lat": lat + float(self.rng.normal(0, 0.009)),
                    "lon": lon + float(self.rng.normal(0, 0.009)),
                    "opinion_continua": opinion,
                    "capital_social": 0.55,
                    "desconfianza": 0.25,
                    "nivel_movilizacion": 0.7,
                }
            )
        df = pd.DataFrame(rows)
        df = self._finalize_calib_df(df)
        adj = self._build_adj_from_alcaldias(df, p_intra=0.18, p_inter=0.06)
        return self._finish_calib(df, adj)

    def _scenario_equilibrio_fragil(self, n=300, intensidad=0.7):
        rows = []
        for i in range(n):
            alc = ALCALDIAS_CDMX[i % 10]
            lat, lon = ALCALDIA_COORDS[alc]
            r = float(self.rng.random())
            if r < 0.34:
                opinion = 0.55
            elif r < 0.67:
                opinion = -0.55
            else:
                opinion = 0.0
            opinion = float(np.clip(opinion + self.rng.normal(0, 0.12), -1, 1))
            rows.append(
                {
                    "seccion": f"E{i:04d}",
                    "alcaldia": alc,
                    "lat": lat + float(self.rng.normal(0, 0.01)),
                    "lon": lon + float(self.rng.normal(0, 0.01)),
                    "opinion_continua": opinion,
                    "capital_social": 0.5,
                    "desconfianza": 0.45,
                    "nivel_movilizacion": 0.5,
                }
            )
        df = pd.DataFrame(rows)
        df = self._finalize_calib_df(df)
        adj = self._build_adj_from_alcaldias(df, p_intra=0.08, p_inter=0.025)
        return self._finish_calib(df, adj)

    def _scenario_broker_dominante(self, n=220, intensidad=0.9):
        df, adj = self._scenario_equilibrio_fragil(n=n, intensidad=0.5)
        hub_idx = len(df)
        alc = "CUAUHTEMOC"
        lat, lon = ALCALDIA_COORDS[alc]
        nuevo = {
            "seccion": "BROKER01",
            "alcaldia": alc,
            "lat": lat,
            "lon": lon,
            "opinion_continua": 0.8,
            "capital_social": 0.95,
            "desconfianza": 0.1,
            "nivel_movilizacion": 0.9,
            "territorial_unit_id": "BROKER01",
            "agent_id": "BROKER-CALIB",
            "spin": 1,
            "intencion": "SIMPATIZANTE",
            "es_broker_insertado": True,
            "habilidades_sociales": 0.93,
            "fatiga": 0.0,
            "capital_politico": 0.95,
            "grado": 0,
            "influencia_SAF": 0.0,
            "acceso_informacion": 0.9,
            "influencia_liderazgo": 0.9,
            "arraigo": 0.8,
            "exposicion_problema": 0.3,
            "resistencia_institucional": 0.2,
            "temperatura_sintetica": 40.0,
            "polarizacion_local": 0.3,
            "calibrado": True,
        }
        df = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
        candidatos = list(range(hub_idx))
        sel = self.rng.choice(
            candidatos, size=min(35, len(candidatos)), replace=False
        )
        for s in sel:
            adj.setdefault(hub_idx, []).append(int(s))
            adj.setdefault(int(s), []).append(hub_idx)
        return self._finish_calib(df, adj)

    # Helpers
    def _finalize_calib_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["territorial_unit_id"] = df["seccion"]
        df["agent_id"] = "CAL-" + df["seccion"]
        df["spin"] = np.where(
            df["opinion_continua"] > 0.25,
            1,
            np.where(df["opinion_continua"] < -0.25, -1, 0),
        )
        df["intencion"] = np.where(
            df["spin"] == 1,
            "SIMPATIZANTE",
            np.where(df["spin"] == -1, "OPOSITOR", "INDECISO"),
        )
        return self._fill_required_columns(df)

    def _finish_calib(self, df: pd.DataFrame, adj: dict) -> Tuple[pd.DataFrame, dict]:
        df = df.reset_index(drop=True)
        df["grado"] = df.index.map(lambda i: len(adj.get(i, [])))
        max_g = max(int(df["grado"].max()), 1)
        df["influencia_SAF"] = (
            df["habilidades_sociales"] * 0.6 + (df["grado"] / max_g) * 0.4
        ).clip(0, 1)
        return df, dict(adj)

    def _fill_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        defaults = {
            "capital_social": 0.5,
            "acceso_informacion": 0.5,
            "influencia_liderazgo": 0.4,
            "arraigo": 0.6,
            "nivel_movilizacion": 0.55,
            "desconfianza": 0.35,
            "exposicion_problema": 0.4,
            "resistencia_institucional": 0.3,
            "temperatura_sintetica": 55.0,
            "fatiga": 0.0,
            "capital_politico": 0.7,
            "es_broker_insertado": False,
            "es_adversario": False,
            "polarizacion_local": 0.4,
            "calibrado": False,
            "habilidades_sociales": 0.5,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        w = self.cfg.get("habilidades", CONFIG["habilidades"])
        df["habilidades_sociales"] = (
            df["capital_social"] * w["capital_social"]
            + df["acceso_informacion"] * w["acceso_informacion"]
            + df["influencia_liderazgo"] * w["influencia_liderazgo"]
            + df["arraigo"] * w["arraigo"]
            + df["nivel_movilizacion"] * w["nivel_movilizacion"]
            + df["desconfianza"] * w["desconfianza"]
        ).clip(0, 1)
        return df

    def _build_adj_from_alcaldias(
        self, df: pd.DataFrame, p_intra: float = 0.08, p_inter: float = 0.02
    ) -> dict:
        adj = defaultdict(list)
        for alc in df["alcaldia"].unique():
            idxs = df.index[df["alcaldia"] == alc].tolist()
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    if float(self.rng.random()) < p_intra:
                        x, y = idxs[a], idxs[b]
                        adj[x].append(y)
                        adj[y].append(x)
        for a1, vecinos in CONTIGUIDAD_ALCALDIAS.items():
            idxs1 = df.index[df["alcaldia"] == a1].tolist()
            for a2 in vecinos:
                idxs2 = df.index[df["alcaldia"] == a2].tolist()
                for x in idxs1:
                    for y in idxs2:
                        if float(self.rng.random()) < p_inter:
                            adj[x].append(y)
                            adj[y].append(x)
        return adj


# =============================================================================
# SECCIÓN 2 · MOTOR ABM Y SIMULACIÓN MONTE CARLO
# =============================================================================


def step_abm(df: pd.DataFrame, adj: dict, beta: float = 1.2) -> pd.DataFrame:
    df_next = df.copy()
    n = len(df)
    spins = df["spin"].values
    influencias = df["influencia_SAF"].values

    for i in range(n):
        vecinos = adj.get(i, [])
        if not vecinos:
            continue
        h_local = sum(influencias[v] * spins[v] for v in vecinos)
        p_cambio = 1.0 / (1.0 + np.exp(-beta * h_local))

        if np.random.rand() < p_cambio:
            nuevo_spin = 1 if h_local >= 0 else -1
        else:
            nuevo_spin = spins[i]

        df_next.at[i, "spin"] = nuevo_spin
        df_next.at[i, "opinion_continua"] = np.clip(
            df_next.at[i, "opinion_continua"] + 0.1 * nuevo_spin, -1, 1
        )

    df_next["intencion"] = np.where(
        df_next["spin"] == 1,
        "SIMPATIZANTE",
        np.where(df_next["spin"] == -1, "OPOSITOR", "INDECISO"),
    )
    return df_next


def run_simulation(
    df_init: pd.DataFrame, adj: dict, beta: float, steps: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_curr = df_init.copy()
    history = []

    for t in range(steps + 1):
        counts = df_curr["spin"].value_counts()
        history.append(
            {
                "step": t,
                "SIMPATIZANTE": counts.get(1, 0),
                "OPOSITOR": counts.get(-1, 0),
                "INDECISO": counts.get(0, 0),
            }
        )
        if t < steps:
            df_curr = step_abm(df_curr, adj, beta=beta)

    return df_curr, pd.DataFrame(history)


# =============================================================================
# SECCIÓN 3 · INTERFAZ PRINCIPAL DE STREAMLIT
# =============================================================================

st.set_page_config(page_title="SITER-CAE v5.0", layout="wide")
st.title("🌐 SITER-CAE v5.0 — Simulador de Dinámica Social y Electoral CDMX")

# ----- SIDEBAR -----
st.sidebar.header("⚙️ Configuración del Universo")

data_mode = st.sidebar.selectbox(
    "Modo de datos",
    options=["synth_pure", "dummy", "synth_coherent", "synth_calib", "real"],
    index=0,
)

seed = st.sidebar.number_input("Seed", 1, 99999, 42)
n = st.sidebar.slider("N nodos / secciones", 50, 2000, 400, 50)

p_intra = 0.05
p_inter = 0.01
scenario = "polarizacion_alta"
intensidad = 0.8
secciones_file = None
shp_file = None
electoral_file = None
socio_file = None
seguridad_file = None

if data_mode == "real":
    st.sidebar.markdown("**Archivos reales (CSV/SHP)**")
    secciones_file = st.sidebar.file_uploader(
        "Catálogo secciones (CSV)", type=["csv"]
    )
    shp_file = st.sidebar.file_uploader(
        "SHP / GeoJSON (opcional)", type=["shp", "geojson", "zip"]
    )
    electoral_file = st.sidebar.file_uploader(
        "Resultados electorales (CSV)", type=["csv"]
    )
    socio_file = st.sidebar.file_uploader(
        "Socio / Marginación (CSV opcional)", type=["csv"]
    )
    seguridad_file = st.sidebar.file_uploader(
        "Seguridad (CSV opcional)", type=["csv"]
    )
    p_intra = st.sidebar.slider("Prob intra-alcaldía", 0.01, 0.20, 0.08, 0.01)
    p_inter = st.sidebar.slider("Prob inter-alcaldía", 0.001, 0.08, 0.02, 0.001)

elif data_mode == "synth_calib":
    scenario = st.sidebar.selectbox(
        "Escenario de calibración",
        [
            "polarizacion_alta",
            "spof_critico",
            "cascade_rapida",
            "equilibrio_fragil",
            "broker_dominante",
        ],
    )
    intensidad = st.sidebar.slider(
        "Intensidad del escenario", 0.3, 1.0, 0.8, 0.05
    )

elif data_mode in ("synth_pure", "synth_coherent", "dummy"):
    p_intra = st.sidebar.slider("Prob intra", 0.01, 0.20, 0.05, 0.01)
    p_inter = st.sidebar.slider("Prob inter", 0.001, 0.08, 0.01, 0.001)

st.sidebar.header("🔬 Parámetros de Simulación")
beta = st.sidebar.slider("Beta influencia", 0.3, 2.5, 1.2, 0.1)
steps = st.sidebar.slider("Pasos ABM", 5, 50, 15)

if st.sidebar.button("🧬 Cargar / Generar universo", type="primary"):
    provider = DataProvider(mode=data_mode, seed=seed, cfg=CONFIG)
    try:
        if data_mode == "real":
            if secciones_file is None:
                st.sidebar.error("Modo real requiere al menos el CSV de secciones")
            else:
                df, adj, meta = provider.load(
                    secciones_csv=secciones_file,
                    shp_path=shp_file,
                    electoral_csv=electoral_file,
                    socio_csv=socio_file,
                    seguridad_csv=seguridad_file,
                    p_intra=p_intra,
                    p_inter=p_inter,
                )
        elif data_mode == "synth_calib":
            df, adj, meta = provider.load(
                scenario=scenario, n=n, intensidad=intensidad
            )
        elif data_mode == "dummy":
            df, adj, meta = provider.load(n_secciones=n)
        elif data_mode == "synth_coherent":
            df, adj, meta = provider.load(n=n)
        else:
            df, adj, meta = provider.load(
                n=n, p_intra=p_intra, p_inter=p_inter
            )

        st.session_state.update(
            {
                "df": df,
                "adj": adj,
                "df_base": df.copy(),
                "adj_base": {k: list(v) for k, v in adj.items()},
                "tray": pd.DataFrame(),
                "df_final": pd.DataFrame(),
                "data_meta": meta,
            }
        )
        st.success(f"Universo inicializado correctamente | Modo: {meta['mode']}")
    except Exception as e:
        st.error(f"Error al cargar universo: {e}")

if st.sidebar.button("♻️ Restaurar estado inicial"):
    if "df_base" in st.session_state and not st.session_state["df_base"].empty:
        st.session_state["df"] = st.session_state["df_base"].copy()
        st.session_state["adj"] = {
            k: list(v) for k, v in st.session_state["adj_base"].items()
        }
        st.session_state["tray"] = pd.DataFrame()
        st.session_state["df_final"] = pd.DataFrame()
        st.success("Universo restaurado al estado base.")

# ----- PANEL PRINCIPAL -----
if "df" in st.session_state and not st.session_state["df"].empty:
    df = st.session_state["df"]
    adj = st.session_state["adj"]
    meta = st.session_state.get("data_meta", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodos totales", len(df))
    col2.metric("Simpatizantes", int((df["spin"] == 1).sum()))
    col3.metric("Opositores", int((df["spin"] == -1).sum()))
    col4.metric("Indecisos", int((df["spin"] == 0).sum()))

    st.markdown("---")

    if st.button("▶️ Ejecutar Simulación ABM"):
        df_final, tray = run_simulation(df, adj, beta=beta, steps=steps)
        st.session_state["df_final"] = df_final
        st.session_state["tray"] = tray

        exp = Experiment(
            df=df, adj=adj, data_meta=meta, beta=beta, steps=steps
        )
        st.session_state["experiment_payload"] = exp.to_payload(df_final, tray)

    if (
        "tray" in st.session_state
        and not st.session_state["tray"].empty
    ):
        st.subheader("📈 Trayectoria de Opinión")
        st.line_chart(
            st.session_state["tray"].set_index("step")[
                ["SIMPATIZANTE", "OPOSITOR", "INDECISO"]
            ]
        )

        st.subheader("📊 Datos del Estado Final")
        st.dataframe(st.session_state["df_final"].head(10))

        st.subheader("📋 JSON Payload (Procedencia y Experimento)")
        st.json(st.session_state["experiment_payload"])
else:
    st.info("Presiona **🧬 Cargar / Generar universo** para comenzar.")
