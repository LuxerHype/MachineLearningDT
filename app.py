from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Cargar el modelo de IA entrenado
MODEL_PATH = 'model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/')
def home():
    return "API de IA para Gestión de Proyectos de Hidrógeno"

# Endpoint para predecir riesgos en proyectos de hidrógeno
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 500
    
    data = request.get_json()
    try:
        # Convertir los datos en un array de numpy para el modelo
        input_data = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint para actualizar el modelo
@app.route('/update_model', methods=['POST'])
def update_model():
    global model
    data = request.get_json()
    try:
        # Aquí podrías entrenar el modelo con nuevos datos
        X_new = np.array(data['features'])
        y_new = np.array(data['labels'])

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_new, y_new)
        joblib.dump(model, MODEL_PATH)
        
        return jsonify({'message': 'Modelo actualizado correctamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()
