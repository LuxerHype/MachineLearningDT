from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# Ejemplo de entrenamiento con 10 características por muestra
# Cada fila representa: [corriente, temperatura, presión, eficiencia, voltajeTotal,
# hidrógeno/s, tempAmbiente, coefConvectivo, resistenciaInterna, numCeldas]
X = np.array([
    [20.0, 50.0, 1.0, 0.9, 2.2, 0.0015, 25.0, 5.0, 0.02, 10],
    [25.0, 60.0, 1.2, 0.85, 2.6, 0.0020, 25.0, 5.0, 0.025, 10],
    [15.0, 45.0, 1.0, 0.95, 2.0, 0.0012, 25.0, 5.0, 0.018, 10],
    [30.0, 70.0, 1.3, 0.8, 2.9, 0.0025, 25.0, 5.0, 0.03, 10]
])
# Etiquetas: 0 = sin riesgo, 1 = riesgo
y = np.array([0, 1, 0, 1])

# Entrenar el modelo
model = LogisticRegression()
model.fit(X, y)

# Guardar el modelo
joblib.dump(model, 'model.pkl')