"""Entrena el modelo (`mlp` o `fisico`, según config.json) sobre el dataset sintético.
Pérdida = NLL gaussiana sobre el espectro + peso_spf * MSE(log SPF calculado del espectro predicho)."""
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fisica import RAIZ, Metricas, Simulador, cargar_json, elegir_dispositivo
from modelo import ModeloFisico, construir_modelo


def perdida(modelo, met, c, v, y, log_spf, peso_spf):
    mu, logvar = modelo(c, v)
    nll = 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar)).mean()
    l_spf = (torch.log(met.spf(mu)) - log_spf).pow(2).mean()
    return nll + peso_spf * l_spf, nll, l_spf


@torch.no_grad()
def evaluar(modelo, met, c, v, y, log_spf, peso_spf, bs=8192):
    modelo.eval()
    tot, mae, preds = 0.0, 0.0, []
    for i in range(0, len(c), bs):
        sl = slice(i, i + bs)
        p, _, _ = perdida(modelo, met, c[sl], v[sl], y[sl], log_spf[sl], peso_spf)
        mu, _ = modelo(c[sl], v[sl])
        n = len(c[sl])
        tot += p.item() * n
        mae += (mu - y[sl]).abs().mean().item() * n
        preds.append(torch.log(met.spf(mu)))
    pred = torch.cat(preds)
    err_rel = (torch.exp(pred - log_spf) - 1).abs()
    r2 = 1 - ((pred - log_spf) ** 2).sum() / ((log_spf - log_spf.mean()) ** 2).sum()
    return tot / len(c), mae / len(c), err_rel.median().item(), r2.item(), pred


def principal():
    cfg = cargar_json("config.json")
    e = cfg["entrenamiento"]
    arq = e["arquitectura"]
    device = elegir_dispositivo(cfg)
    torch.manual_seed(0)

    datos = torch.load(RAIZ / cfg["rutas"]["datos"], weights_only=False)
    c = datos["c"].to(device)
    v = datos["vehiculo"].to(device)
    y = datos["A_medida"].to(device)
    log_spf = torch.log(datos["spf"]).to(device)

    sim = Simulador(device, cfg)
    if sim.nombres != datos["nombres"]:
        raise RuntimeError("filtros.json cambio desde que generaste los datos: vuelve a correr 1_generar_datos.bat")
    met = Metricas(sim.l, cfg["simulacion"].get("espectro_solar_csv"))

    n = len(c)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(device)
    nval = int(n * e["fraccion_validacion"])
    iv, it = perm[:nval], perm[nval:]

    modelo = construir_modelo(arq, sim.F, len(sim.vehiculos), len(sim.l), sim.max_pct, cfg).to(device)
    n_par = sum(p.numel() for p in modelo.parameters())
    print(f"Arquitectura: {arq} | parametros: {n_par:,} | device: {device} | train {len(it)} / val {nval}")

    bs, epocas = e["batch"], e["epocas"]
    pasos = epocas * math.ceil(len(it) / bs)
    opt = torch.optim.AdamW(modelo.parameters(), lr=e["lr"][arq], weight_decay=e["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=e["lr"][arq], total_steps=pasos, pct_start=0.05)

    destino = RAIZ / cfg["rutas"]["modelo"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    mejor = float("inf")
    historial = []
    it_eval = it[:min(20000, len(it))]  # subconjunto fijo de train para medirlo con las mismas métricas que val
    ruta_hist = destino.parent / f"historial_{arq}.json"
    for ep in range(1, epocas + 1):
        modelo.train()
        t0 = time.time()
        orden = it[torch.randperm(len(it), device=device)]
        acum = torch.zeros((), device=device)
        for i in range(0, len(orden), bs):
            idx = orden[i:i + bs]
            p, _, _ = perdida(modelo, met, c[idx], v[idx], y[idx], log_spf[idx], e["peso_spf"])
            opt.zero_grad(set_to_none=True)
            p.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            sched.step()
            acum += p.detach() * len(idx)
        tr, tr_mae, tr_err, tr_r2, _ = evaluar(modelo, met, c[it_eval], v[it_eval], y[it_eval], log_spf[it_eval], e["peso_spf"])
        val, mae, err_med, r2, _ = evaluar(modelo, met, c[iv], v[iv], y[iv], log_spf[iv], e["peso_spf"])
        historial.append({
            "epoca": ep, "loss_train_lotes": (acum / len(orden)).item(),
            "loss_train": tr, "loss_val": val, "mae_train": tr_mae, "mae_val": mae,
            "err_spf_train": tr_err, "err_spf_val": err_med, "r2_train": tr_r2, "r2_val": r2,
            "lr": sched.get_last_lr()[0], "segundos": time.time() - t0,
        })
        with open(ruta_hist, "w", encoding="utf-8") as fh:
            json.dump(historial, fh, indent=1)
        marca = ""
        if val < mejor:
            mejor = val
            torch.save({"estado": modelo.state_dict(), "arquitectura": arq, "config": cfg,
                        "nombres": sim.nombres, "vehiculos": sim.vehiculos, "nval": nval,
                        "n_parametros": n_par, "mejor_epoca": ep}, destino)
            marca = " *"
        print(f"ep {ep:3d} | train {tr:8.4f} | val {val:8.4f} | MAE espectro {mae:.4f} | error SPF mediano "
              f"{100 * err_med:5.1f}% | R2 logSPF {tr_r2:.4f}/{r2:.4f} | {time.time() - t0:4.1f}s{marca}")
    print(f"Historial -> {ruta_hist}")

    ck = torch.load(destino, weights_only=False)
    modelo.load_state_dict(ck["estado"])
    graficar(cfg, modelo, sim, met, c[iv], v[iv], y[iv], log_spf[iv], datos["A_efectiva"].to(device)[iv], e["peso_spf"])


@torch.no_grad()
def graficar(cfg, modelo, sim, met, c, v, y, log_spf, a_limpia, peso_spf):
    graf = RAIZ / cfg["rutas"]["graficas"]
    graf.mkdir(parents=True, exist_ok=True)
    arq = cfg["entrenamiento"]["arquitectura"]
    _, _, _, r2, pred = evaluar(modelo, met, c, v, y, log_spf, peso_spf)
    l = sim.l.cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    m = min(20000, len(pred))
    ax[0].scatter(torch.exp(log_spf[:m]).cpu(), torch.exp(pred[:m]).cpu(), s=1, alpha=0.3)
    lim = [1, float(torch.exp(log_spf).max())]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("SPF real (simulador)"); ax[0].set_ylabel("SPF predicho")
    ax[0].set_title(f"{arq}: R2 log SPF = {r2:.4f}")
    mu, logvar = modelo(c[:4], v[:4])
    sd = torch.exp(0.5 * logvar)
    for j in range(4):
        linea, = ax[1].plot(l, mu[j].cpu(), lw=1.5)
        ax[1].fill_between(l, (mu[j] - 2 * sd[j]).cpu(), (mu[j] + 2 * sd[j]).cpu(), color=linea.get_color(), alpha=0.15)
        ax[1].plot(l, a_limpia[j].cpu(), "--", color=linea.get_color(), lw=1)
        ax[1].scatter(l[::4], y[j, ::4].cpu(), s=4, color=linea.get_color())
    ax[1].set_xlabel("nm"); ax[1].set_ylabel("Absorbancia efectiva")
    ax[1].set_title("Prediccion ±2σ, verdad (--), medicion (puntos)")
    fig.tight_layout(); fig.savefig(graf / f"entrenamiento_{arq}.png", dpi=120)

    if isinstance(modelo, ModeloFisico):
        dosis = cfg["simulacion"]["dosis_uv"]
        aprendido = modelo.absorbancia_por_pct() * torch.exp(-modelo.fotolabilidad() * dosis).unsqueeze(-1)
        verdad = cfg["simulacion"]["espesor_cm"] * sim.E * torch.exp(-sim.k_foto * dosis).unsqueeze(-1)
        filas = math.ceil(sim.F / 4)
        fig, axs = plt.subplots(filas, 4, figsize=(14, 3 * filas))
        for i, a in enumerate(axs.flat):
            if i >= sim.F:
                a.axis("off"); continue
            a.plot(l, verdad[i].cpu(), "k--", lw=1, label="verdad")
            a.plot(l, aprendido[i].cpu(), lw=1.5, label="aprendido")
            a.set_title(sim.nombres[i], fontsize=9)
        axs.flat[0].legend(fontsize=7)
        fig.suptitle("Absorbancia por 1% tras fotodegradacion (sin interacciones)")
        fig.tight_layout(); fig.savefig(graf / "espectros_aprendidos.png", dpi=120)
        k = torch.nn.functional.softplus(modelo.p_k).tolist()
        print("Forma de pelicula aprendida (k):", {n: round(x, 3) for n, x in zip(sim.vehiculos, k)},
              "| verdad:", cfg["vehiculos"])
    print(f"Graficas -> {graf}")


if __name__ == "__main__":
    principal()
