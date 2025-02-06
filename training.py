from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# Datos de ejemplo para entrenamiento: cada muestra tiene 2 características
X = np.array([
    [0.5, 0.2],
    [0.9, 0.8],
    [0.3, 0.5]
])
# Etiquetas correspondientes a cada muestra
y = np.array([0, 1, 0])  # 0 = Sin riesgo, 1 = Riesgo

# Entrenar el modelo
model = LogisticRegression()
model.fit(X, y)

# Guardar el modelo entrenado en un archivo .pkl
joblib.dump(model, 'model.pkl')