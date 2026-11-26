# Mejora individual del MVP — modelo D con EfficientNet-B0

**Autor:** Marcos Agustín García Mombello

El modelo D va a clasificar radiografías de tórax PA según la etiqueta `Effusion` de NIH ChestX-ray14. El experimento toma como referencia el modelo B, que fue el modelo final del MVP, y cambia DenseNet121 por EfficientNet-B0. La pregunta es si una red con menos parámetros puede mantener o mejorar el rendimiento de B sobre las mismas imágenes.

## Datos y condiciones de la comparación

Se va a usar la carpeta `data/processed/nih_effusion_pa_modelo_b_v1`, verificada contra los hashes originales del equipo. Contiene 22.594 imágenes distintas de entrenamiento, 1.813 de validación y 4.390 de test, separadas por paciente. D mantiene la resolución de 224 × 224, el preprocesamiento, los aumentos, la repetición de positivos, la semilla 42 y los hiperparámetros de B. La inicialización usa los pesos de ImageNet correspondientes a EfficientNet-B0.

## Qué se va a realizar

1. Agregar `efficientnet_b0` a `build_model` en `src/rx_torax/training/model.py`. Se van a cargar los pesos de ImageNet de `torchvision` y reemplazar la última capa del clasificador por una salida para `Effusion`.
2. Crear `configs/modelo_b_efficientnet_b0.json` a partir de `configs/modelo_b.json`, con `efficientnet_b0` como arquitectura y `models/modelo_d_efficientnet_b0` como salida.
3. Hacer una corrida corta para verificar que D entrena, guarda un `modelo.pt` y puede volver a cargarlo para predecir. Adaptar el mapa de calor de `rx-probar` al flujo de EfficientNet-B0 y verificar que su puntaje coincida con el de `rx-predecir` para la misma imagen.
4. Entrenar D con los datos completos. Elegir la época por AUC y el umbral por F1, ambos en validación. Ejecutar `rx-comparar` con las predicciones de D y las publicadas de B para estimar la diferencia pareada por paciente.
5. Con la época y el umbral fijados, evaluar D una sola vez en test y comparar sus predicciones con las que ya están publicadas para B. Guardar la configuración, el historial, el checkpoint, las predicciones y los reportes.

## Qué esperamos que cambie

Con la salida binaria, D tiene **4.008.829 parámetros** frente a los **6.954.881** de B, una reducción del **42,4 %**. La hipótesis es que esa reducción puede mantener un rendimiento cercano al de B. El efecto sobre F1 y AUC se medirá con el experimento.

| Métrica | B en validación | B en test | Qué se va a comprobar en D |
| --- | ---: | ---: | --- |
| F1 de `Effusion` | 0,800 | 0,744 | Es la métrica principal. Se medirá si D mantiene o mejora el equilibrio entre precisión y recall con su umbral de validación. |
| AUC ROC | 0,905 | 0,869 | Se medirá si D ordena mejor las imágenes positivas y negativas a lo largo de todos los umbrales. |
| Recall | 85,5 % | 93,1 % | Se revisará la proporción de positivos detectados. |
| Especificidad | 81,2 % | 61,8 % | Se revisará la proporción de negativos correctamente descartados. |

Se informará la diferencia **D − B** y su intervalo de confianza del 95 % para F1 y AUC. Un intervalo completamente por encima de cero respaldaría una mejora en esa métrica. Si incluye cero, la diferencia queda inconclusa con esta evaluación. También se registrarán el dispositivo y las versiones usadas para interpretar los resultados.

Los valores de referencia salen de los reportes del modelo B. [Validación](models/modelo_b_densenet121_v1/evaluacion_val/report.json) · [Test](models/modelo_b_densenet121_v1/evaluacion_test/report.json) · [Protocolo del equipo](docs/MEJORAS_FINAL.md#protocolo)

## Entrenamiento en una GTX 1080

Para la GTX 1080 (Pascal) en Windows, usar Python 3.12 y PyTorch 2.14.1 con CUDA 12.6 en un entorno separado. PyTorch 2.14 es la última versión que publica binarios para esta arquitectura. [Compatibilidad de PyTorch con Pascal](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)

Desde la raíz del repositorio, creá el entorno, instalá las versiones CUDA y verificá que PyTorch vea la GPU.

```powershell
$envCuda = Join-Path $env:LOCALAPPDATA 'venvs\iaproyecto-cu126'
python -m venv $envCuda
$pythonCuda = Join-Path $envCuda 'Scripts\python.exe'
& $pythonCuda -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126 torchvision==0.29.1+cu126
& $pythonCuda -m pip install -e .
& $pythonCuda -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
& $pythonCuda -m rx_torax.training.train --config configs/modelo_b_efficientnet_b0.json
```

El reporte de D registra el dispositivo y las versiones de las bibliotecas. Esos datos acompañan la comparación con B porque el entorno de entrenamiento puede influir en los resultados.
