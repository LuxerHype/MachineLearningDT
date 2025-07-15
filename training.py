from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import joblib

# 1. Datos de ejemplo ampliados (idealmente reemplázalos con datos reales)
#    Aquí generamos 100 muestras sintéticas para ilustrar.
np.random.seed(42)
n_samples = 100

corriente = np.random.uniform(10, 30, n_samples)
temperatura = np.random.uniform(40, 80, n_samples)
presion = np.random.uniform(0.8, 1.4, n_samples)
eficiencia = np.random.uniform(0.75, 0.98, n_samples)
voltajeTotal = np.random.uniform(1.8, 3.0, n_samples)
produccion = np.random.uniform(0.0005, 0.003, n_samples)
tempAmb = np.full(n_samples, 25.0)
coefConv = np.full(n_samples, 5.0)
resInterna = np.random.uniform(0.015, 0.03, n_samples)
numCeldas = np.full(n_samples, 10.0)

# Construir matriz X de forma (n_samples, 10)
X = np.vstack([
    corriente,
    temperatura,
    presion,
    eficiencia,
    voltajeTotal,
    produccion,
    tempAmb,
    coefConv,
    resInterna,
    numCeldas
]).T

# 2. Etiquetas sintéticas: definimos riesgo=1 cuando corriente y temp altas + baja eficiencia
y = ((corriente > 25) & (temperatura > 65) & (eficiencia < 0.85)).astype(int)

# 3. Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Crear pipeline con escalado y modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

# 5. Entrenamiento con validación cruzada
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Accuracy CV (5‑fold): {scores.mean():.3f} ± {scores.std():.3f}")

# 6. Fit final y evaluación en test
pipeline.fit(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"Accuracy en test: {test_score:.3f}")

# 7. Guardar pipeline completo
joblib.dump(pipeline, 'model.pkl')
print("Modelo entrenado y guardado como model.pkl")