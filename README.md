#Predicción de Calidad de Vino con Red Neuronal Artificial (RNA)
Este proyecto utiliza una Red Neuronal Artificial (RNA) para predecir si un vino es de alta calidad (puntuación igual o superior a 7) o baja calidad (puntuación inferior a 7) basándose en sus características fisicoquímicas.

Contenido
wine_quality.csv: El dataset utilizado para entrenar y evaluar el modelo (asegúrate de incluirlo en tu repositorio).
modelo_vino.ipynb (o .py): El código fuente con la implementación completa del modelo, incluyendo comentarios detallados.
README.md: Este archivo, que proporciona una descripción general del proyecto y instrucciones de uso.
Requisitos
Python 3.x
Bibliotecas:
numpy
pandas
scikit-learn
Keras (con TensorFlow como backend)
Puedes instalar las dependencias utilizando:

Bash
pip install numpy pandas scikit-learn keras tensorflow
Usa el código con precaución.
content_copy
Cómo usar el modelo
Clona el repositorio:

Bash
git clone https://github.com/tu_usuario/tu_repositorio.git
Usa el código con precaución.
content_copy
Ejecuta el código:

Si tienes el código en un archivo .ipynb (Jupyter Notebook), ábrelo en Jupyter y ejecuta todas las celdas.
Si tienes el código en un archivo .py, ejecútalo desde la terminal:
Bash
python modelo_vino.py
Usa el código con precaución.
content_copy
Interpreta los resultados:

El script mostrará la matriz de confusión y la precisión del modelo en los datos de prueba.
También realizará predicciones para dos nuevos ejemplos de vino.
Estructura del modelo
Entrada: 11 características fisicoquímicas del vino.
Capas ocultas: Dos capas densas con 6 neuronas cada una, utilizando la función de activación ReLU.
Capa de salida: Una neurona con activación sigmoide para la predicción binaria (alta/baja calidad).
Dataset
El dataset wine_quality.csv contiene información sobre vinos portugueses, incluyendo características como acidez fija, acidez volátil, ácido cítrico, azúcar residual, cloruros, dióxido de azufre libre, dióxido de azufre total, densidad, pH, sulfatos y alcohol. La variable objetivo es la calidad del vino (0-10), que se ha binarizado para este proyecto.

Consideraciones
El modelo ha sido entrenado con un conjunto de datos específico. Su rendimiento podría variar con otros tipos de vino.
Experimenta con diferentes arquitecturas de red neuronal y parámetros para optimizar el rendimiento.
