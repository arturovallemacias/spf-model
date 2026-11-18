"""Dos arquitecturas:
- mlp:    caja negra residual (~0.3M parámetros) que predice el espectro efectivo y su varianza.
- fisico: el esqueleto de la física con todo aprendido (~2k parámetros interpretables):
          absorbancia por % de cada filtro, fotolabilidad, matriz de interacciones y forma de película por vehículo.
`forward` conserva su nombre porque PyTorch lo exige.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

LN10 = math.log(10.0)


class Bloque(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norma = nn.LayerNorm(d)
        self.l1 = nn.Linear(d, 2 * d)
        self.l2 = nn.Linear(2 * d, d)

    def forward(self, x):
        return x + self.l2(Fn.silu(self.l1(self.norma(x))))


class ModeloMLP(nn.Module):
    def __init__(self, F, V, L, max_pct, ancho=128, capas=4):
        super().__init__()
        self.register_buffer("max_pct", max_pct.clone())
        self.V = V
        self.entrada = nn.Linear(F + V, ancho)
        self.bloques = nn.Sequential(*[Bloque(ancho) for _ in range(capas)])
        self.salida = nn.Sequential(nn.LayerNorm(ancho), nn.Linear(ancho, 2 * L))

    def forward(self, c, veh):
        x = torch.cat([c / self.max_pct, Fn.one_hot(veh, self.V).float()], -1)
        mu, logvar = self.salida(self.bloques(self.entrada(x))).chunk(2, -1)
        return Fn.softplus(mu), logvar.clamp(-10, 5)


class ModeloFisico(nn.Module):
    def __init__(self, F, V, L, max_pct, dosis=1.0):
        super().__init__()
        self.register_buffer("max_pct", max_pct.clone())
        self.dosis = dosis
        self.p_abs = nn.Parameter(torch.full((F, L), -1.0))   # absorbancia por 1% en película (softplus)
        self.p_foto = nn.Parameter(torch.full((F,), -3.0))    # fotolabilidad (softplus)
        self.W = nn.Parameter(torch.zeros(F, F))               # interacciones objetivo x agente
        self.p_k = nn.Parameter(torch.zeros(V))                # forma de la película por vehículo (softplus)
        self.logvar = nn.Parameter(torch.full((V, L), -4.0))   # ruido aleatorio por vehículo y longitud de onda

    def absorbancia_por_pct(self):
        return Fn.softplus(self.p_abs)

    def fotolabilidad(self):
        return Fn.softplus(self.p_foto)

    def forward(self, c, veh):
        cn = c / self.max_pct
        m = torch.exp((cn @ self.W.T).clamp(-10, 10))
        c_post = c * torch.exp(-self.fotolabilidad() * self.dosis * m)
        A = c_post @ self.absorbancia_por_pct()
        k = (Fn.softplus(self.p_k)[veh] + 1e-3).unsqueeze(-1)
        A_eff = k * torch.log10(1.0 + A * LN10 / k)
        return A_eff, self.logvar[veh]


def construir_modelo(arquitectura, F, V, L, max_pct, cfg):
    e = cfg["entrenamiento"]
    if arquitectura == "mlp":
        return ModeloMLP(F, V, L, max_pct, ancho=e["ancho"], capas=e["capas"])
    if arquitectura == "fisico":
        return ModeloFisico(F, V, L, max_pct, dosis=cfg["simulacion"]["dosis_uv"])
    raise ValueError(f"Arquitectura desconocida: {arquitectura} (usa 'mlp' o 'fisico')")
