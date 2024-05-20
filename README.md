# 🍷 Predicción de Calidad de Vino con Red Neuronal Artificial (RNA) 🍷

Este proyecto aprovecha el poder de una **Red Neuronal Artificial (RNA)** para predecir si un vino es de **alta calidad** (puntuación ≥ 7) o **baja calidad** (puntuación < 7) a partir de sus características fisicoquímicas.

## 🚀 Contenido

- `wine_quality.csv`: El dataset 🍇 utilizado para entrenar y evaluar el modelo.
- `modelo_vino.ipynb` (o `.py`): El código fuente 🧠 con la implementación completa y comentarios detallados.
- `README.md`: Este archivo que estás leyendo ahora mismo, ¡tu guía para entender y usar el proyecto!

## 🛠️ Requisitos

- Python 3.x
- Bibliotecas:
    - numpy
    - pandas
    - scikit-learn
    - Keras (con TensorFlow como backend)

Instala todo fácilmente con:

```bash
pip install numpy pandas scikit-learn keras tensorflow
```

## 🏃 Cómo usar el modelo

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/itsOrlo/modeloPredictivoVinos.git
   ```

2. **¡A ejecutar!**
   - **Jupyter Notebook (`modelo_vino.ipynb`)**: Ábrelo en Jupyter y ejecuta todas las celdas.
   - **Script Python (`modelo_vino.py`)**:
     ```bash
     python modelo_vino.py
     ```

3. **Interpreta los resultados:**
   - Matriz de confusión: Compara predicciones vs. realidad.
   - Precisión: Qué tan bien lo hizo el modelo en general.
   - Predicciones: ¡Prueba con nuevos vinos!

## 🧠 Estructura del modelo

- **Entrada:** 11 características del vino (acidez, azúcar, etc.).
- **Capas ocultas:** 2 capas densas (6 neuronas c/u) con activación ReLU.
- **Capa de salida:** 1 neurona con activación sigmoide (0 = baja calidad, 1 = alta calidad).

## 🍇 Dataset

El dataset `wine_quality.csv` contiene información sobre vinos portugueses 🇵🇹, incluyendo:

- Acidez fija, volátil, cítrica
- Azúcar residual
- Cloruros, dióxido de azufre
- Densidad, pH, sulfatos
- Alcohol
- ¡Y la calidad (0-10) que queremos predecir!

## ⚠️ Consideraciones

- El modelo se entrenó con vinos portugueses. ¡Pruébalo con otras regiones!
- Experimenta cambiando la arquitectura de la red o los parámetros. ¡Siempre hay margen de mejora!


