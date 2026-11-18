"""Simulador físico aproximado de protectores solares (gemelo sintético).

Cadena: concentraciones -> fotodegradación -> absorbancia uniforme (Beer-Lambert)
-> película irregular (espesor ~ Gamma) -> absorbancia efectiva -> SPF, UVA-PF, lambda crítica.
Todos los espectros son aproximaciones: reemplázalos por datos medidos cuando los tengas.
"""
import json
import math
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent
LN10 = math.log(10.0)


def cargar_json(nombre):
    with open(RAIZ / nombre, encoding="utf-8") as f:
        return json.load(f)


def elegir_dispositivo(cfg):
    pedido = cfg.get("device", "cuda")
    if pedido.startswith("cuda") and not torch.cuda.is_available():
        print("AVISO: CUDA no disponible, usando CPU.")
        return torch.device("cpu")
    return torch.device(pedido)


def rejilla(device):
    return torch.arange(290, 401, dtype=torch.float32, device=device)  # 111 longitudes de onda (290-400 nm)


def espectro_eritema(l):
    """Espectro de acción eritemal CIE (1998)."""
    medio = 10.0 ** (0.094 * (298.0 - l))
    alto = 10.0 ** (0.015 * (140.0 - l))
    return torch.where(l <= 298, torch.ones_like(l), torch.where(l <= 328, medio, alto))


def espectro_ppd(l):
    """Aproximación del espectro de acción PPD (UVA): ~1.4 décadas entre 320 y 400 nm."""
    p = 10.0 ** (-(l - 320.0) / 57.0)
    return torch.where(l >= 320, p, torch.zeros_like(l))


def espectro_solar(l, csv=None):
    """Irradiancia solar relativa aproximada. csv opcional con columnas lambda,valor (con encabezado)."""
    if csv:
        import numpy as np
        d = np.loadtxt(RAIZ / csv, delimiter=",", skiprows=1)
        v = np.interp(l.cpu().numpy(), d[:, 0], d[:, 1])
        return torch.tensor(v, dtype=torch.float32, device=l.device)
    bajo = 0.35 * 10.0 ** (-0.12 * (320.0 - l))
    alto = 0.35 + 0.65 * (l - 320.0) / 80.0
    return torch.where(l < 320, bajo, alto)


def absorbancia_efectiva(A, k):
    """Película con espesor relativo h ~ Gamma(k, 1/k):
    T = E[10^(-A h)] = (1 + A ln10 / k)^(-k)  =>  A_eff = k log10(1 + A ln10 / k)."""
    k = k.unsqueeze(-1)
    return k * torch.log10(1.0 + A * LN10 / k)


class Metricas:
    """SPF, UVA-PF y lambda crítica a partir de la absorbancia efectiva [..., 111]. Diferenciable (salvo lambda_c)."""

    def __init__(self, l, csv=None):
        sol = espectro_solar(l, csv)
        self.l = l
        self.w_spf = espectro_eritema(l) * sol
        self.w_uva = espectro_ppd(l) * sol

    def spf(self, A_eff):
        T = 10.0 ** (-A_eff)
        return self.w_spf.sum() / (T * self.w_spf).sum(-1)

    def uvapf(self, A_eff):
        T = 10.0 ** (-A_eff)
        return self.w_uva.sum() / (T * self.w_uva).sum(-1)

    def lambda_critica(self, A_eff):
        acum = torch.cumsum(A_eff, -1)
        total = acum[..., -1:]
        idx = (acum >= 0.9 * total).float().argmax(-1)
        return self.l[idx]


class Simulador:
    def __init__(self, device, cfg=None, filtros=None):
        self.cfg = cfg or cargar_json("config.json")
        fdat = filtros or cargar_json("filtros.json")
        self.device = device
        self.l = rejilla(device)
        lista = fdat["filtros"]
        self.nombres = [f["id"] for f in lista]
        self.F = len(lista)
        self.max_pct = torch.tensor([f["max_pct"] for f in lista], dtype=torch.float32, device=device)
        self.costo = torch.tensor([f.get("costo_rel", 1.0) for f in lista], dtype=torch.float32, device=device)
        self.k_foto = torch.tensor([f.get("fotolabilidad", 0.0) for f in lista], dtype=torch.float32, device=device)
        self.E = torch.stack([self._espectro_filtro(f) for f in lista])  # [F, L], E 1% 1cm

        idx = {n: i for i, n in enumerate(self.nombres)}
        self.W = torch.zeros(self.F, self.F, device=device)  # fila: objetivo, columna: agente
        for it in fdat.get("interacciones", []):
            s = it["fuerza"] if it["tipo"] == "desestabiliza" else -it["fuerza"]
            self.W[idx[it["objetivo"]], idx[it["agente"]]] = s

        self.vehiculos = list(self.cfg["vehiculos"].keys())
        self.k_veh = torch.tensor(list(self.cfg["vehiculos"].values()), dtype=torch.float32, device=device)

    def _espectro_filtro(self, f):
        l = self.l
        if f.get("tipo") == "inorganico":
            return f["E_meseta"] * torch.sigmoid((f["borde_nm"] - l) / f["ancho_borde"])
        E = torch.zeros_like(l)
        for p in f["picos"]:
            E = E + p["E"] * torch.exp(-0.5 * ((l - p["nm"]) / p["sigma"]) ** 2)
        return E

    def fotodegradar(self, c, dosis):
        cn = c / self.max_pct
        m = torch.exp((cn @ self.W.T).clamp(-10, 10))
        return c * torch.exp(-self.k_foto * dosis * m)

    def absorbancia_uniforme(self, c):
        return self.cfg["simulacion"]["espesor_cm"] * (c @ self.E)

    def simular(self, c, veh, ruido=True):
        """c: [B, F] en %, veh: [B] índices. Devuelve (A_medida, A_efectiva) [B, 111]."""
        s = self.cfg["simulacion"]
        k = self.k_veh[veh]
        if ruido:
            k = k * torch.exp(s["jitter_pelicula"] * torch.randn_like(k))
        c_post = self.fotodegradar(c, s["dosis_uv"])
        A_eff = absorbancia_efectiva(self.absorbancia_uniforme(c_post), k)
        if not ruido:
            return A_eff, A_eff
        A_med = A_eff * (1 + s["ruido_rel"] * torch.randn_like(A_eff)) + s["ruido_abs"] * torch.randn_like(A_eff)
        return A_med.clamp_min(0.0), A_eff

    def vector_formula(self, filtros_pct):
        c = torch.zeros(self.F, device=self.device)
        for nombre, pct in filtros_pct.items():
            c[self.nombres.index(nombre)] = pct
        return c
