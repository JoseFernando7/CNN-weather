# Clasificación de Fenómenos Climáticos Usando Redes Neuronales Convolucionales (CNN)

## Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar un modelo de **Clasificación de Imágenes** que identifique diferentes fenómenos climáticos a partir de un conjunto de datos de imágenes. Algunos de los fenómenos considerados incluyen lluvia, nieve, arcoíris, tormentas de arena, y otros. La aplicación de este modelo podría ser útil en campos como meteorología, investigación ambiental, y sistemas de monitoreo automáticos.

El enfoque principal del proyecto se basó en utilizar redes neuronales convolucionales (**CNNs**) para aprovechar las capacidades de estas redes en el procesamiento de imágenes.

## Estructura del Proyecto

### Datos
Se utilizó un conjunto de datos de imágenes clasificadas en diferentes fenómenos climáticos. Las imágenes fueron divididas en:
- **Entrenamiento**: Para ajustar los pesos del modelo.
- **Validación**: Para ajustar hiperparámetros y prevenir el sobreajuste.
- **Prueba**: Para evaluar el rendimiento del modelo.

### Arquitectura del Modelo
El modelo se diseñó utilizando la librería **Keras** en TensorFlow. La arquitectura principal incluye:
1. **Capas Convolucionales**:
   - Cuatro capas convolucionales con filtros de tamaño (3x3), seguidas de **Batch Normalization** y **MaxPooling**.
   - Se eligieron múltiples capas convolucionales para capturar jerarquías de características (desde bordes simples hasta patrones complejos).

2. **Capas Densas**:
   - Dos capas densas para combinar las características extraídas y realizar la clasificación.
   - Uso de funciones de activación `ReLU` para no linealidad y `softmax` en la capa de salida.

3. **Batch Normalization**:
   - Normalización de los datos dentro de cada mini-batch para estabilizar el aprendizaje y acelerar la convergencia.

4. **Dropout**:
   - Técnica de regularización para prevenir el sobreajuste.

### Etapas del Desarrollo
1. **Preprocesamiento de Datos**:
   - Redimensionamiento de imágenes a (64x64).
   - Normalización de los valores de los píxeles al rango [0, 1].

2. **Diseño y Entrenamiento del Modelo**:
   - Arquitectura de red basada en CNN.
   - Optimizador Adam, función de pérdida `categorical_crossentropy`.
   - Entrenamiento en 15 épocas con un tamaño de lote (“batch size”) de 32.

3. **Evaluación**:
   - Métricas utilizadas: Precisión (accuracy), métricas de clasificación (precision, recall, F1-score).
   - Visualización del rendimiento a través de matrices de confusión y curvas de aprendizaje.

### Resultados Obtenidos
- **Precisión del Modelo**: Aproximadamente 66% en el conjunto de prueba.
- **Reporte de Clasificación**:
   - Variación en el rendimiento entre clases: Fenómenos como "rime" y "lightning" obtuvieron altos valores de recall, mientras que "frost" y "hail" tuvieron menores desempeños.

- **Matriz de Confusión**:
   - Visualización detallada de las predicciones correctas e incorrectas para cada clase.

### Tecnologías Utilizadas
- **Python**
- **TensorFlow/Keras**
- **Matplotlib y Seaborn** (para visualizaciones)
- **NumPy y Pandas** (para manipulación de datos)

## Instrucciones para Reproducir
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/JoseFernando7/CNN-weather
   cd CNN-weather
   ```

## Lecciones Aprendidas y Futuro Trabajo
### Lecciones Aprendidas
- **Importancia del Preprocesamiento**: La calidad de las imágenes y el preprocesamiento impactan directamente el rendimiento del modelo.
- **Batch Normalization y Dropout**: Fueron esenciales para estabilizar el entrenamiento y mejorar la generalización.

### Futuras Mejoras
1. Incorporar técnicas de aumento de datos (data augmentation) para mejorar la robustez del modelo.
2. Experimentar con arquitecturas pre-entrenadas (como ResNet o EfficientNet) usando transferencia de aprendizaje.
3. Optimizar hiperparámetros (tasa de aprendizaje, número de filtros, etc.) usando herramientas como Grid Search o Bayesian Optimization.

---

Este proyecto es un ejemplo de cómo las CNN pueden resolver problemas complejos de clasificación de imágenes y es una base para futuros desarrollos en el área de aprendizaje profundo aplicado a fenómenos climáticos.

