"""Genera el dataset sintético: formulaciones aleatorias -> espectros, SPF, UVA-PF, lambda crítica."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fisica import RAIZ, Metricas, Simulador, cargar_json, elegir_dispositivo


def muestrear(sim, n, s, device):
    nf = torch.randint(s["min_filtros"], min(s["max_filtros"], sim.F) + 1, (n,), device=device)
    rango = torch.rand(n, sim.F, device=device).argsort(-1).argsort(-1)
    mascara = (rango < nf.unsqueeze(-1)).float()
    c = (0.1 + 0.9 * torch.rand(n, sim.F, device=device)) * sim.max_pct * mascara
    carga = c.sum(-1, keepdim=True).clamp_min(1e-6)
    c = c * torch.clamp(s["carga_max_pct"] / carga, max=1.0)
    veh = torch.randint(0, len(sim.vehiculos), (n,), device=device)
    return c, veh


@torch.no_grad()
def principal():
    cfg = cargar_json("config.json")
    s = cfg["simulacion"]
    device = elegir_dispositivo(cfg)
    torch.manual_seed(s["semilla"])
    sim = Simulador(device, cfg)
    met = Metricas(sim.l, s.get("espectro_solar_csv"))

    print("Formulas de referencia (sin ruido) - usalas para calibrar el simulador:")
    for ref in cfg.get("formulas_referencia", []):
        c = sim.vector_formula(ref["filtros"]).unsqueeze(0)
        v = torch.tensor([sim.vehiculos.index(ref["vehiculo"])], device=device)
        _, a = sim.simular(c, v, ruido=False)
        spf, uva = met.spf(a).item(), met.uvapf(a).item()
        lc = met.lambda_critica(a).item()
        print(f"  {ref['nombre']:<45} SPF {spf:6.1f}  UVA-PF {uva:5.1f}  UVA/SPF {uva / spf:4.2f}  lambda_c {lc:.0f} nm")

    C, V, AM, AE = [], [], [], []
    lote = 50000
    for i in range(0, s["n_muestras"], lote):
        n = min(lote, s["n_muestras"] - i)
        c, v = muestrear(sim, n, s, device)
        a_med, a_eff = sim.simular(c, v, ruido=True)
        C.append(c); V.append(v); AM.append(a_med); AE.append(a_eff)
    c, v, a_med, a_eff = torch.cat(C), torch.cat(V), torch.cat(AM), torch.cat(AE)
    spf, uva, lc = met.spf(a_eff), met.uvapf(a_eff), met.lambda_critica(a_eff)

    datos = {
        "c": c.cpu(), "vehiculo": v.cpu(), "A_medida": a_med.cpu(), "A_efectiva": a_eff.cpu(),
        "spf": spf.cpu(), "uvapf": uva.cpu(), "lambda_c": lc.cpu(),
        "nombres": sim.nombres, "vehiculos": sim.vehiculos,
    }
    destino = RAIZ / cfg["rutas"]["datos"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    torch.save(datos, destino)

    q = torch.quantile(spf, torch.tensor([0.05, 0.5, 0.95], device=device)).tolist()
    print(f"\n{len(c)} muestras -> {destino}")
    print(f"SPF p5 {q[0]:.1f} | mediana {q[1]:.1f} | p95 {q[2]:.1f}")

    graf = RAIZ / cfg["rutas"]["graficas"]
    graf.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(torch.log10(spf).cpu().numpy(), bins=80, color="#4a7")
    ax[0].set_xlabel("log10 SPF"); ax[0].set_title("Distribucion de SPF")
    l = sim.l.cpu().numpy()
    for j in range(6):
        ax[1].plot(l, a_med[j].cpu().numpy(), lw=0.8, alpha=0.6)
        ax[1].plot(l, a_eff[j].cpu().numpy(), lw=1.5, label=f"SPF {spf[j].item():.0f}")
    ax[1].set_xlabel("nm"); ax[1].set_ylabel("Absorbancia efectiva"); ax[1].legend(fontsize=8)
    ax[1].set_title("Ejemplos (fino = medido con ruido)")
    fig.tight_layout(); fig.savefig(graf / "datos.png", dpi=120)
    print(f"Grafica -> {graf / 'datos.png'}")


if __name__ == "__main__":
    principal()
