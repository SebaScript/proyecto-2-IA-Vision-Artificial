# Proyecto 2 IA: Clasificación de gatos y perros con fine tuning de VGG16 y ResNet50
## Juan Sebastian García Perez

Clasificador capaz de distinguir automáticamente si en una fotografía aparece un gato o un perro, aplicando fine tuning sobre redes convolucionales preentrenadas con ImageNet.

**Dominio:** Visión artificial, clasificación de imágenes de animales domésticos.

**Pregunta:** Dada la fotografía de una mascota, ¿es un gato o un perro?

**Tipo de problema:** Aprendizaje supervisado - clasificación binaria de imágenes.

**Utilidad:** Automatizar el etiquetado de fotos de mascotas. Un refugio o un portal de adopción recibe miles de fotos y necesita clasificarlas por especie sin que una persona las revise una por una.

## BD seleccionada:

Cats vs Dogs, creada por Microsoft Research y Petfinder.com. No está en el repositorio: se carga desde TensorFlow Datasets en una celda del notebook. https://www.tensorflow.org/datasets/catalog/cats_vs_dogs

| Característica | Valor |
|---|---|
| Imágenes | 23.262 fotografías JPEG a color |
| Variable de entrada | image: matriz de píxeles de alto x ancho x 3 canales (RGB), enteros de 0 a 255 |
| Variable de salida | label (binaria): cat -> 0, dog -> 1 |
| Tamaño de las imágenes | Variable, entre 4 y 500 píxeles de lado |
| Reparto de clases | 11.658 gatos (50,1 %), 11.604 perros (49,9 %) |
| Particiones | 70 % entrenamiento (16.283), 15 % validación (3.490), 15 % prueba (3.489) |

## Metodología:

1. **Análisis exploratorio**: estructura de una imagen y sus canales, ejemplos de cada clase, reparto de clases y tamaño de las imágenes.
2. **Preparación**: redimensionado de todas las imágenes a 160 x 160 y agrupación en lotes. La normalización de cada red va dentro del modelo.
3. **Modelos**: base convolucional de VGG16 y de ResNet50 preentrenadas con ImageNet, más una cabeza nueva (GlobalAveragePooling, Dense, Dropout y salida sigmoide).
4. **Fine tuning en dos fases**: primero se entrena solo la cabeza con la base congelada y después se descongela el último bloque convolucional (block5 en VGG16, conv5 en ResNet50) con una tasa de aprendizaje de 0,00001.
5. **Malla de hiperparámetros**: 8 combinaciones por arquitectura, 16 entrenamientos en total, sobre una muestra de 5.000 imágenes.
6. **Comparación y evaluación**: la mejor combinación de cada arquitectura se reentrena con todos los datos y el ganador se evalúa sobre el conjunto de prueba.

### Hiperparámetros explorados

| Hiperparámetro | Valores |
|---|---|
| Arquitectura | VGG16, ResNet50 |
| Unidades de la capa densa | 64, 256 |
| Tasa de aprendizaje de la cabeza | 0,001, 0,0001 |
| Tamaño del lote | 32, 64 |

## Resultados

Mejor combinación de cada arquitectura en la malla y resultado al reentrenar con todos los datos (exactitud en validación):

| Modelo | Unidades | lr | batch | Malla | Solo cabeza | Con fine tuning |
|---|---|---|---|---|---|---|
| **ResNet50** | 256 | 0,0001 | 32 | 0,9811 | 0,9814 | **0,9840** |
| VGG16 | 64 | 0,001 | 32 | 0,9794 | 0,9777 | 0,9822 |

**Modelo elegido: ResNet50 con fine tuning del bloque conv5.**

Sobre el conjunto de prueba (3.489 imágenes que no intervinieron en ninguna decisión):

| Métrica | gato | perro |
|---|---|---|
| Precisión | 0,9741 | 0,9857 |
| Sensibilidad | 0,9855 | 0,9746 |
| F1 | 0,9798 | 0,9801 |

Exactitud: **0,9799**, con 70 errores de 3.489 imágenes (2,0 %).

### Conclusiones:

1. La transferencia de aprendizaje funciona muy bien en este problema: con la base congelada y solo 5.000 imágenes ya se alcanza entre 94 % y 98 % de exactitud.

2. El fine tuning mejora los resultados de forma consistente: en la malla mejoró 15 de las 16 combinaciones y en los modelos finales sumó entre 0,3 y 0,5 puntos.

3. Los modelos son poco sensibles a los hiperparámetros: las 16 combinaciones quedan entre 97,2 % y 98,1 %. Lo que más influyó fue la arquitectura.

4. ResNet50 fue mejor que VGG16 en la malla y en el entrenamiento final, aunque por poco (6 imágenes de diferencia en validación).

5. No hay un sobreajuste preocupante, ya que la exactitud en test (97,99 %) es casi igual a la de validación (98,40 %).

## Limitaciones

- El sistema solo distingue estas dos clases, por lo que ante la foto de otro animal respondería igualmente gato o perro.
- La malla se evaluó con una muestra de 5.000 imágenes y una sola partición de validación, y las diferencias entre combinaciones son tan pequeñas que parte de ellas puede deberse al azar.
- Las imágenes se redujeron a 160 x 160 y no se usó aumento de datos.

## Cómo ejecutarlo

Abrir `Proyecto_Vision_Cats_vs_Dogs.ipynb` en un entorno con GPU (por ejemplo Google Colab) y ejecutar todas las celdas. La primera vez descarga la BD (unos 790 MB). Necesita tensorflow, tensorflow-datasets, pandas, matplotlib, seaborn y scikit-learn.

Los resultados de este repositorio se obtuvieron con TensorFlow 2.19 y una GPU RTX 4050 en unos 35 minutos. Al volver a ejecutarlo las cifras pueden cambiar en la tercera decimal y, como las dos arquitecturas quedan tan cerca, también cuál de las dos gana.
