# Parte 1 - Preparación de los datos

# Importar las librerías necesarias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # Para dividir datos
from sklearn.preprocessing import StandardScaler      # Para escalar datos
from keras.models import Sequential                  # Para crear la RNA
from keras.layers import Dense                       # Para las capas de la RNA
from sklearn.metrics import confusion_matrix, accuracy_score  # Para evaluar

# Cargar el dataset desde un archivo CSV
dataset = pd.read_csv('wine_quality.csv')

# Separar las características (X) y la variable objetivo (y)
X = dataset.iloc[:, :-1].values  # Todas las columnas excepto la última
y = dataset.iloc[:, -1].values   # La última columna (calidad del vino)

# Binarizar la variable objetivo: 0 si calidad < 7, 1 si calidad >= 7
y = np.where(y >= 7, 1, 0)

# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalar las características para mejorar el rendimiento del modelo
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Ajustar y transformar datos de entrenamiento
X_test = sc.transform(X_test)        # Transformar datos de prueba (sin ajustar)


# Parte 2 - Construcción de la RNA

# Inicializar la RNA como una secuencia de capas
classifier = Sequential()

# Capa de entrada: 11 neuronas (11 características del vino), activación ReLU
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Capa oculta: 6 neuronas, activación ReLU
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Capa de salida: 1 neurona (predicción binaria), activación sigmoide
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compilar la RNA: optimizador Adam, función de pérdida para clasificación binaria
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar la RNA con los datos de entrenamiento (100 iteraciones, lotes de 10)
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# Parte 3 - Evaluación y predicciones

# Predecir la calidad del vino para los datos de prueba
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)  # Convertir probabilidades a etiquetas binarias (0 o 1)

# Mostrar la matriz de confusión (comparación entre predicciones y valores reales)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calcular y mostrar la precisión del modelo en los datos de prueba
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Hacer predicciones con nuevos datos (escalados previamente)
new_prediction = classifier.predict(sc.transform(np.array([[10.80, 1.16, 0.60, 8.86, 0.27, 46.86, 129.84, 1.0022, 3.96, 0.97, 13.55]])))
new_prediction = (new_prediction > 0.5)
print(new_prediction)

new_prediction = classifier.predict(sc.transform(np.array([[9.31, 1.17, 0.00, 5.31, 0.10, 7.56, 58.71, 0.9948, 3.24, 1.23, 11.1]]))) 
new_prediction = (new_prediction > 0.5)
print(new_prediction)