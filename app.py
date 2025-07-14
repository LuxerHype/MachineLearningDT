from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = 'model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.route('/')
def home():
    return "API de IA para Evaluación de Riesgos de Electrólisis PEM"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 500
    
    data = request.get_json()
    try:
        features = data['features']
        input_array = np.array(features).reshape(1, -1)

        prediction = model.predict(input_array)
        return jsonify({'risk': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/update_model', methods=['POST'])
def update_model():
    global model
    data = request.get_json()
    try:
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
