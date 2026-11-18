# SPF·MODEL — IA que predice cuánta luz del sol deja pasar un protector solar

**[▶ Demo en vivo](https://arturovallemacias.github.io/spf-model/)**: mueve los filtros UV y la IA calcula el SPF al instante, en tu navegador.

<!-- Agrega una captura o GIF del laboratorio virtual en docs/demo.gif y quita este comentario:
![Laboratorio virtual](docs/demo.gif)
-->

## El problema

El SPF de un protector se mide en voluntarios humanos (ISO 24444). Cada prueba tarda semanas y cuesta miles de dólares, y el formulador suele iterar varios prototipos antes de dar con la fórmula. Este proyecto predice el espectro de protección, el SPF y la protección UVA a partir de la receta de filtros y el formato (crema, spray o barra), y además **diseña la receta a la inversa**: tú le das el SPF objetivo y te devuelve la fórmula de menor costo.

## Cómo funciona

1. **Gemelo digital físico** (`fisica.py`): genera 200 mil fórmulas sintéticas con fotodegradación e interacciones entre filtros, Beer-Lambert y una película irregular sobre la piel. Esa película tiene solución cerrada:

   T(λ) = (1 + A·ln10/k)^(−k)  ⇒  A_eff = k·log10(1 + A·ln10/k),  con espesor h ~ Gamma(k, 1/k)

   Por eso el SPF real queda muy por debajo del teórico, y el spray (k pequeña) protege menos que la crema.
2. **Dos modelos en PyTorch** (`modelo.py`):

   | | `mlp` | `fisico` |
   |---|---|---|
   | Parámetros | ~0.3M | ~2k, cada uno con significado físico |
   | Aprende | el espectro efectivo y su incertidumbre | espectro por filtro, fotolabilidad, interacciones y forma de la película |

3. **Pérdida:** NLL gaussiana sobre el espectro más el error en log SPF, calculado de forma diferenciable con los espectros de acción eritemal (CIE) y PPD.
4. **Diseño inverso** (`disenar.py`): optimización por gradiente a través del modelo, con restricciones de SPF, balance UVA/UVB ≥ 1/3, carga máxima y costo. Cada receta se verifica con el simulador.
5. **Panel interactivo** (`docs/`): la IA corre en el navegador con los pesos exportados. Incluye un laboratorio con sliders, un mapa 3D (filtro × formato → protección), los resultados train vs val y un glosario.

## Resultados

| Métrica (validación) | `mlp` | `fisico` |
|---|---|---|
| R² log SPF | 0.895 | — |
| Error mediano de SPF | 12.5 % | — |
| Brecha val / train | ~1.0 (sin sobreajuste) | — |

El 12.5 % es el **piso de ruido físico**: la irregularidad de la película varía ±20 % entre aplicaciones y el modelo no la observa. Si se quita ese ruido (`jitter_pelicula = 0`), el error baja a pocas unidades porcentuales. Es un orden de magnitud realista: el mismo producto medido in vivo en distintos laboratorios varía de forma comparable.

## Uso local (Windows)

1. `instalar.bat`: crea el venv e instala PyTorch con CUDA.
2. `1_generar_datos.bat`: genera el dataset sintético.
3. `2_entrenar.bat`: entrena (elige `mlp` o `fisico` en `config.json`).
4. `3_disenar.bat`: diseño inverso hacia el SPF objetivo.
5. `4_panel.bat`: genera el panel local y la versión web en `docs/`.

Todas las entradas se configuran en `config.json` y `filtros.json`.

## Limitaciones

- Está entrenado con un gemelo digital, no con mediciones reales. **No sustituye la prueba oficial de SPF.**
- Los espectros, los límites legales y la fotolabilidad de `filtros.json` son aproximados. Verifica los límites vigentes (COFEPRIS, FDA, UE) antes de usarlos.
- Siguiente paso: afinar el modelo `fisico` con unas decenas de espectros in vitro reales (placas de PMMA). Con su estructura física, pocos datos bastan.

---
Hecho por Arturo · León, México
