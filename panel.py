"""Genera el panel: laboratorio virtual, mapa 3D y resultados del modelo (train vs val) frente a las mediciones.
Lee modelos/modelo.pt, modelos/historial_<arq>.json y datos/dataset.pt. Abre el panel en el navegador."""
import base64
import json
import webbrowser

import torch

from entrenar import evaluar
from fisica import RAIZ, Metricas, Simulador, cargar_json, elegir_dispositivo
from modelo import construir_modelo


def redondear(x, nd=4):
    return [round(float(v), nd) for v in x]


@torch.no_grad()
def principal():
    cfg = cargar_json("config.json")
    device = elegir_dispositivo(cfg)
    ck = torch.load(RAIZ / cfg["rutas"]["modelo"], map_location=device, weights_only=False)
    arq = ck["arquitectura"]
    ccfg = ck["config"]
    datos = torch.load(RAIZ / ccfg["rutas"]["datos"], weights_only=False)
    with open(RAIZ / "modelos" / f"historial_{arq}.json", encoding="utf-8") as fh:
        historial = json.load(fh)

    sim = Simulador(device, ccfg)
    met = Metricas(sim.l, ccfg["simulacion"].get("espectro_solar_csv"))
    modelo = construir_modelo(arq, sim.F, len(sim.vehiculos), len(sim.l), sim.max_pct, ccfg).to(device)
    modelo.load_state_dict(ck["estado"])
    modelo.eval()

    c = datos["c"].to(device)
    v = datos["vehiculo"].to(device)
    y = datos["A_medida"].to(device)
    a_limpia = datos["A_efectiva"].to(device)
    log_spf = torch.log(datos["spf"]).to(device)
    n = len(c)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(device)
    nval = ck.get("nval", int(n * ccfg["entrenamiento"]["fraccion_validacion"]))
    iv, it = perm[:nval], perm[nval:]
    it_eval = it[:min(20000, len(it))]
    peso = ccfg["entrenamiento"]["peso_spf"]

    def resumen(idx):
        loss, mae, err, r2, pred = evaluar(modelo, met, c[idx], v[idx], y[idx], log_spf[idx], peso)
        cob = []
        for i in range(0, len(idx), 8192):
            sl = idx[i:i + 8192]
            mu, logvar = modelo(c[sl], v[sl])
            sd = torch.exp(0.5 * logvar)
            cob.append(((y[sl] - mu).abs() <= 2 * sd).float().mean(-1))
        return {"loss": loss, "mae": mae, "err_spf": err, "r2": r2,
                "cobertura_2s": torch.cat(cob).mean().item()}, pred

    m_train, pred_tr = resumen(it_eval)
    m_val, pred_val = resumen(iv)

    # error por vehículo (val)
    por_veh = []
    for k, nombre in enumerate(sim.vehiculos):
        msk = v[iv] == k
        e = (torch.exp(pred_val[msk] - log_spf[iv][msk]) - 1).abs().median().item()
        por_veh.append({"vehiculo": nombre, "err_spf": e, "n": int(msk.sum())})

    # dispersión: predicho vs medido
    ns = 2500
    disp = {
        "train_real": redondear(log_spf[it_eval][:ns].exp(), 2), "train_pred": redondear(pred_tr[:ns].exp(), 2),
        "val_real": redondear(log_spf[iv][:ns].exp(), 2), "val_pred": redondear(pred_val[:ns].exp(), 2),
    }

    # histogramas de residuos log(pred/real)
    bordes = torch.linspace(-0.3, 0.3, 32, device=device)
    def histograma(res):
        h = torch.histc(res.clamp(-0.3, 0.2999), bins=31, min=-0.3, max=0.3)
        return redondear(h / h.sum(), 5)
    residuos = {"bordes": redondear(bordes, 3),
                "train": histograma(pred_tr - log_spf[it_eval]), "val": histograma(pred_val - log_spf[iv])}

    # muestras de val para el paisaje 3D, ordenadas por SPF
    m = 48
    elegidos = iv[torch.linspace(0, len(iv) - 1, 400, device=device).long()]
    elegidos = elegidos[log_spf[elegidos].argsort()][torch.linspace(0, 399, m, device=device).long()]
    mu, logvar = modelo(c[elegidos], v[elegidos])
    sd = torch.exp(0.5 * logvar)
    muestras = []
    for j, i in enumerate(elegidos.tolist()):
        receta = {nom: round(x, 2) for nom, x in zip(sim.nombres, c[i].tolist()) if x > 0}
        muestras.append({
            "mu": redondear(mu[j], 3), "sd": redondear(sd[j], 3), "medido": redondear(y[i], 3), "limpio": redondear(a_limpia[i], 3),
            "spf_real": round(log_spf[i].exp().item(), 1), "spf_pred": round(met.spf(mu[j]).item(), 1),
            "uva_pred": round(met.uvapf(mu[j]).item(), 1),
            "vehiculo": sim.vehiculos[int(v[i])], "receta": receta,
        })

    # pesos del modelo para correr la inferencia en el navegador (float32 little-endian en base64)
    pesos = {k: {"shape": list(t.shape),
                 "data": base64.b64encode(t.detach().float().cpu().contiguous().numpy().astype("<f4").tobytes()).decode()}
             for k, t in modelo.state_dict().items()}
    fdat = cargar_json("filtros.json")
    frac_uva = (sim.E[:, sim.l >= 320].sum(1) / sim.E.sum(1)).tolist()
    filtros = []
    for f, u in zip(fdat["filtros"], frac_uva):
        banda = "Mineral" if f.get("tipo") == "inorganico" else ("UVB" if u < 0.3 else "UVA" if u > 0.7 else "Amplio")
        filtros.append({"id": f["id"], "nombre": f["nombre"], "max": f["max_pct"], "banda": banda})

    D = {
        "pesos": pesos, "filtros": filtros, "dosis": ccfg["simulacion"]["dosis_uv"],
        "w_spf": [float(x) for x in met.w_spf], "w_uva": [float(x) for x in met.w_uva],
        "referencias": ccfg.get("formulas_referencia", []),
        "carga_max_entreno": ccfg["simulacion"]["carga_max_pct"],
        "vehiculos": sim.vehiculos,
        "arquitectura": arq, "n_parametros": ck.get("n_parametros"), "mejor_epoca": ck.get("mejor_epoca"),
        "n_train": len(it), "n_val": nval, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "lambdas": list(range(290, 401)), "historial": historial,
        "metricas": {"train": m_train, "val": m_val}, "por_vehiculo": por_veh,
        "dispersion": disp, "residuos": residuos, "muestras": muestras,
    }

    plantilla = (RAIZ / "panel_plantilla.html").read_text(encoding="utf-8")
    fondo = RAIZ / "fondo.jpg"  # imagen de contexto del encabezado (opcional)
    uri = "data:image/jpeg;base64," + base64.b64encode(fondo.read_bytes()).decode() if fondo.exists() else ""
    plantilla = plantilla.replace("/*__FONDO__*/", uri)
    html = plantilla.replace("/*__DATOS__*/null", json.dumps(D, ensure_ascii=False, separators=(",", ":")))
    destino = RAIZ / cfg["rutas"]["graficas"] / f"panel_{arq}.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(html, encoding="utf-8")
    print(f"Panel -> {destino}")

    # versión web (GitHub Pages): los pesos van en un archivo aparte que el panel descarga
    web = RAIZ / "docs"
    web.mkdir(exist_ok=True)
    D_web = {k: v for k, v in D.items() if k != "pesos"}
    D_web["pesos_url"] = "pesos.json"
    (web / "index.html").write_text(
        plantilla.replace("/*__DATOS__*/null", json.dumps(D_web, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    (web / "pesos.json").write_text(json.dumps(pesos, separators=(",", ":")), encoding="utf-8")
    print(f"Version web -> {web} (GitHub Pages publica esta carpeta)")
    webbrowser.open(destino.resolve().as_uri())


if __name__ == "__main__":
    principal()
