# Mejora individual del MVP con EfficientNet-B0

**Autor** — Marcos Agustín García Mombello

Probé **EfficientNet-B0** para detectar derrame pleural (Effusion) en radiografías de tórax PA del NIH. El **Modelo D** es la mejora principal. El **Modelo E** pone a prueba un cambio en la selección de radiografías negativas de entrenamiento. Los dos parten de pesos de ImageNet y tienen una salida binaria.

## Qué cambió

| Modelo | Arquitectura | Entrenamiento |
| --- | --- | --- |
| **B** | DenseNet121 | Conjunto y receta de referencia del MVP |
| **D** | EfficientNet-B0 | Mismas imágenes y receta de B, con los pesos de ImageNet de EfficientNet-B0 |
| **E** | EfficientNet-B0 | Receta de D con 730 reemplazos de imágenes negativas |

En E, cada reemplazo conserva el paciente y el grupo de hallazgos («sin hallazgos» u «otros hallazgos»). Una placa con menos de 10 estudios previos se reemplaza por otra placa PA sin derrame, con 10 o más estudios previos. Se auditaron los píxeles para evitar duplicados exactos. El cambio abarca **544 pacientes**, con 525 reemplazos sin hallazgos y 205 con otros hallazgos. El [manifiesto de reemplazos](models/modelo_e_efficientnet_b0/reemplazos_train.csv) identifica la imagen original, su sustituta y el hash de píxeles.

D y E usan **22.594 imágenes distintas de entrenamiento**, 30.810 ejemplos por época al repetir las positivas, resolución **224 × 224**, semilla **42** y la misma normalización. Comparten **1.813 radiografías de validación** y **4.390 de test**, idénticas byte por byte. El test es la submuestra PA del listado oficial del NIH que usó B, con un 40 % de positivas. En cada corrida se eligió la época por AUC y el umbral por F1 en validación.

## Resultados

| Modelo | F1 en validación | AUC en validación | F1 en test | AUC en test | Pesos |
| --- | ---: | ---: | ---: | ---: | ---: |
| **B · DenseNet121** | 0,8000 | 0,9051 | 0,7436 | **0,8692** | 28,41 MB |
| **D · EfficientNet-B0** | 0,7937 | 0,8974 | **0,7594** | 0,8682 | 16,32 MB |
| **E · EfficientNet-B0** | **0,8019** | 0,8975 | 0,7415 | 0,8593 | 16,32 MB |

**D mejoró el F1 de B en test en 1,58 puntos porcentuales**, con un IC 95 % pareado por paciente de **+0,68 a +2,49 puntos**. Dio **291 falsas alarmas menos** (714 frente a 1.005), mejoró la especificidad de **61,8 % a 72,9 %** y el recall pasó de **93,1 % a 86,1 %**. La diferencia de AUC fue de −0,0010, con un IC 95 % de −0,0064 a +0,0045. El archivo de pesos ocupa **42,5 % menos** y la red binaria tiene 4.008.829 parámetros frente a 6.954.881 de B.

**E logró F1 de 0,8019 en validación.** El IC 95 % de la diferencia contra D va de −0,77 a +2,27 puntos porcentuales. A igual detección de 554 derrames, E produjo 129 falsas alarmas frente a 117 de D. En el grupo de 10 o más estudios previos, produjo 27 frente a 23. Por ese resultado, D fue el modelo elegido. Se evaluó E en test de forma **exploratoria**, con el umbral ya fijado. Frente a B, E redujo las falsas alarmas de **1.005 a 931** y mantuvo la reducción de tamaño de EfficientNet-B0. Su F1 en test fue 0,7415, con una diferencia contra B de −0,21 puntos porcentuales y un IC 95 % de −1,07 a +0,63 puntos.

La meta de **F1 de 0,80 en test** quedó pendiente. B es la referencia DenseNet121 del MVP para las comparaciones pareadas de validación y test. A obtuvo F1 de 0,7951 en la validación compartida. Los resultados de Control y C corresponden a otra partición. B se entrenó en una Mac M4 con MPS y D/E en una GTX 1080 con CUDA. Los intervalos reflejan la variación de la muestra evaluada, con una corrida por modelo.

| Comparación pareada | Validación | Test |
| --- | --- | --- |
| B–D | [Reporte](models/comparacion_val_b_d/report.json) · [Gráfico](models/comparacion_val_b_d/comparacion_val.png) | [Reporte](models/comparacion_test_b_d/report.json) · [Gráfico](models/comparacion_test_b_d/comparacion_test.png) |
| B–E | [Reporte](models/comparacion_val_b_e/report.json) · [Gráfico](models/comparacion_val_b_e/comparacion_val.png) | [Reporte](models/comparacion_test_b_e/report.json) · [Gráfico](models/comparacion_test_b_e/comparacion_test.png) |
| D–E | [Reporte](models/comparacion_val_d_e/report.json) · [Criterio de selección](models/comparacion_val_d_e/seleccion_test.json) | [Reporte](models/comparacion_test_d_e/report.json) · [Gráfico](models/comparacion_test_d_e/comparacion_test.png) |

## Usar los pesos y verificar las métricas

Desde la raíz del repositorio, instalá el proyecto. Los directorios de [D](models/modelo_d_efficientnet_b0) y [E](models/modelo_e_efficientnet_b0) incluyen los pesos originales, su historial y las predicciones por imagen de validación y test. Con esas predicciones, rx-recalcular reproduce las métricas y los intervalos pareados sin descargar NIH ni usar GPU.

~~~bash
python -m pip install -e .
rx-predecir --modelo models/modelo_d_efficientnet_b0 examples/radiografias/00022899_024.png
rx-predecir --modelo models/modelo_e_efficientnet_b0 examples/radiografias/00022899_024.png
rx-recalcular
~~~

## Reconstruir los datos y reentrenar

Descargá el [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC). Colocá Data_Entry_2017_v2020.csv, train_val_list.txt y test_list.txt juntos. En la subcarpeta images/ dejá los doce archivos images_001.tar.gz a images_012.tar.gz. El script de D usa los [manifiestos publicados](configs/nih_effusion_pa_modelo_b_v1_manifiestos.zip) y coteja los seis arrays contra los SHA-256 originales. El de E comprueba los 730 reemplazos contra el listado publicado. Si ya tenés el conjunto preparado original de B, podés verificarlo con rx-verificar-datos data/processed/nih_effusion_pa_modelo_b_v1 y usarlo directamente como base de E.

Windows con Python 3.12 y GTX 1080. El entorno usa las mismas versiones de las corridas publicadas. [Compatibilidad de PyTorch con Pascal](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)

~~~powershell
$envCuda = Join-Path $env:LOCALAPPDATA 'venvs\iaproyecto-cu126'
py -3.12 -m venv $envCuda
$pythonCuda = Join-Path $envCuda 'Scripts\python.exe'
& $pythonCuda -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126 torchvision==0.29.1+cu126
& $pythonCuda -m pip install -e .
& $pythonCuda -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

$nih = 'data/raw/nih'
$d = 'data/processed/nih_effusion_pa_modelo_b_v1'
$e = 'data/processed/nih_effusion_pa_modelo_e_v1'
& $pythonCuda scripts/preparar_modelo_d.py --nih $nih --salida $d
& $pythonCuda scripts/preparar_modelo_e.py --nih $nih --base $d --salida $e --verificar-reemplazos models/modelo_e_efficientnet_b0/reemplazos_train.csv
& $pythonCuda -m rx_torax.data.checksums $e

& $pythonCuda -m rx_torax.training.train --config configs/modelo_b_efficientnet_b0.json --salida work/reentrenado/modelo_d
& $pythonCuda -m rx_torax.training.train --config configs/modelo_e_efficientnet_b0.json --salida work/reentrenado/modelo_e
~~~

La reconstrucción verifica las imágenes y sus arrays. Los pesos publicados permiten volver a usar los modelos de esta entrega. La semilla y el entorno permiten repetir la receta de entrenamiento. Los pesos de una corrida nueva pueden variar según el hardware y las bibliotecas.

**Datos** — NIH Clinical Center, NIH ChestX-ray14. [Descarga oficial](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [Wang et al., IEEE CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html)
