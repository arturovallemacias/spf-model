"""Diseño inverso: busca recetas de filtros que alcancen el SPF objetivo (con margen), balance UVA,
carga máxima y menor costo. Optimiza por gradiente a través del modelo entrenado y verifica con el simulador."""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as Fn

from fisica import RAIZ, Metricas, Simulador, cargar_json, elegir_dispositivo
from modelo import construir_modelo


def objetivo(c, mu, met, sim, d):
    spf, uva = met.spf(mu), met.uvapf(mu)
    meta = math.log(d["spf_objetivo"] * d["margen_seguridad"])
    cn = c / sim.max_pct
    return ((torch.log(spf) - meta) ** 2
            + d["peso_uva"] * Fn.relu(d["uva_ratio_min"] - uva / spf) ** 2
            + d["peso_carga"] * Fn.relu(c.sum(-1) - d["carga_max_pct"]) ** 2
            + d["peso_costo"] * (c * sim.costo).sum(-1) / 100
            + d["peso_num_filtros"] * (cn / (cn + 0.05)).sum(-1))


def principal():
    cfg = cargar_json("config.json")
    d = cfg["diseno"]
    device = elegir_dispositivo(cfg)
    torch.manual_seed(0)

    ck = torch.load(RAIZ / cfg["rutas"]["modelo"], map_location=device, weights_only=False)
    sim = Simulador(device, cfg)
    if sim.nombres != ck["nombres"]:
        raise RuntimeError("filtros.json no coincide con el modelo entrenado: vuelve a generar y entrenar")
    met = Metricas(sim.l, cfg["simulacion"].get("espectro_solar_csv"))
    modelo = construir_modelo(ck["arquitectura"], sim.F, len(sim.vehiculos), len(sim.l), sim.max_pct, ck["config"]).to(device)
    modelo.load_state_dict(ck["estado"])
    modelo.eval().requires_grad_(False)

    mascara = torch.ones(sim.F, device=device)
    if d["filtros_permitidos"]:
        mascara = torch.tensor([1.0 if n in d["filtros_permitidos"] else 0.0 for n in sim.nombres], device=device)

    R = d["reinicios"]
    veh = torch.full((R,), sim.vehiculos.index(d["vehiculo"]), dtype=torch.long, device=device)
    z = (torch.randn(R, sim.F, device=device) - 1.0).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=d["lr"])
    for paso in range(d["pasos"]):
        c = torch.sigmoid(z) * sim.max_pct * mascara
        mu, _ = modelo(c, veh)
        f = objetivo(c, mu, met, sim, d)
        opt.zero_grad()
        f.sum().backward()
        opt.step()
        if paso % 200 == 0:
            print(f"paso {paso:4d} | objetivo mediano {f.median().item():.4f}")

    with torch.no_grad():
        c = torch.sigmoid(z) * sim.max_pct * mascara
        c = torch.where(c < d["umbral_minimo_pct"], torch.zeros_like(c), c)
        mu, _ = modelo(c, veh)
        f = objetivo(c, mu, met, sim, d)
        _, a_sim = sim.simular(c, veh, ruido=False)
        orden = f.argsort()

        resultados, vistos = [], set()
        for i in orden.tolist():
            clave = tuple(round(x, 1) for x in c[i].tolist())
            if clave in vistos:
                continue
            vistos.add(clave)
            receta = {n: round(x, 2) for n, x in zip(sim.nombres, c[i].tolist()) if x > 0}
            resultados.append({
                "receta_pct": receta,
                "carga_pct": round(c[i].sum().item(), 2),
                "costo_rel": round((c[i] * sim.costo).sum().item() / 100, 3),
                "spf_modelo": round(met.spf(mu[i]).item(), 1),
                "spf_simulador": round(met.spf(a_sim[i]).item(), 1),
                "uva_spf_simulador": round((met.uvapf(a_sim[i]) / met.spf(a_sim[i])).item(), 3),
                "lambda_c_simulador": met.lambda_critica(a_sim[i]).item(),
                "idx": i,
            })
            if len(resultados) >= d["top"]:
                break

    print(f"\nObjetivo: SPF {d['spf_objetivo']} (x{d['margen_seguridad']} margen) en {d['vehiculo']}\n")
    for k, r in enumerate(resultados, 1):
        receta = ", ".join(f"{n} {p}%" for n, p in r["receta_pct"].items())
        print(f"#{k}  {receta}")
        print(f"    carga {r['carga_pct']}% | costo {r['costo_rel']} | SPF modelo {r['spf_modelo']} | "
              f"SPF simulador {r['spf_simulador']} | UVA/SPF {r['uva_spf_simulador']} | lambda_c {r['lambda_c_simulador']:.0f} nm")

    res = RAIZ / cfg["rutas"]["resultados"]
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "diseno.json", "w", encoding="utf-8") as fh:
        json.dump({"diseno": d, "resultados": [{k: v for k, v in r.items() if k != "idx"} for r in resultados]},
                  fh, ensure_ascii=False, indent=2)

    graf = RAIZ / cfg["rutas"]["graficas"]
    graf.mkdir(parents=True, exist_ok=True)
    l = sim.l.cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, r in enumerate(resultados[:3], 1):
        linea, = ax.plot(l, mu[r["idx"]].cpu(), lw=1.8, label=f"#{k} modelo (SPF {r['spf_modelo']})")
        ax.plot(l, a_sim[r["idx"]].cpu(), "--", color=linea.get_color(), lw=1, label=f"#{k} simulador (SPF {r['spf_simulador']})")
    ax.axvline(320, color="gray", lw=0.5)
    ax.set_xlabel("nm"); ax.set_ylabel("Absorbancia efectiva"); ax.legend(fontsize=8)
    ax.set_title(f"Recetas para SPF {d['spf_objetivo']}")
    fig.tight_layout(); fig.savefig(graf / "diseno.png", dpi=120)
    print(f"\nResultados -> {res / 'diseno.json'} | grafica -> {graf / 'diseno.png'}")


if __name__ == "__main__":
    principal()
