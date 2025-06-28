from flask import Flask, render_template, request, redirect, url_for
import pickle
import os
import numpy as np

app = Flask(__name__)

# === Model Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    'diabetes': os.path.join(BASE_DIR, 'Models', 'diabetes_model.sav'),
    'heart': os.path.join(BASE_DIR, 'Models', 'heart_disease_model.sav'),
    'parkinsons': os.path.join(BASE_DIR, 'Models', 'parkinsons_model.sav'),
    'lung': os.path.join(BASE_DIR, 'Models', 'lungs_disease_model.sav'),
    'thyroid': os.path.join(BASE_DIR, 'Models', 'Thyroid_model.sav')
}

# === Feature Definitions ===
FEATURE_CONFIGS = {
    'diabetes': ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree', 'age'],
    'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    'parkinsons': ['fo', 'fhi', 'flo', 'Jitter_percent', 'Jitter_Abs', 'RAP', 'PPQ', 'DDP', 'Shimmer', 'Shimmer_dB',
                   'APQ3', 'APQ5', 'APQ', 'DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'],
    'lung': ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE', 'FATIGUE',
             'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
             'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'],
    'thyroid': ['age', 'sex', 'on_thyroxine', 'tsh', 't3_measured', 't3', 'tt4']
}

# === Load Models ===
def load_models():
    models = {}
    for disease, path in MODEL_PATHS.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[disease] = pickle.load(f)
        else:
            models[disease] = None
    return models

models = load_models()

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose')
def diagnose():
    disease_type = request.args.get('type', '')
    return render_template('diagnose.html', disease_type=disease_type)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        disease_type = request.form.get('disease_type')
        model = models.get(disease_type)

        if not model:
            return render_template('result.html', error="Model not found or not loaded.")

        try:
            features = get_features(request.form, disease_type)
            prediction, confidence = make_prediction(model, np.array(features).reshape(1, -1))

            results = {
                'disease_type': disease_type.title(),
                'prediction': int(prediction),
                'probability': float(confidence),
                'features': {k: v for k, v in request.form.items() if k != 'disease_type'},
                'recommendations': get_recommendations(disease_type, prediction),
                'risk_level': get_risk_level(confidence)
            }

            return render_template('result.html', results=results)

        except Exception as e:
            return render_template('result.html', error=str(e))

    return redirect(url_for('index'))

# === Helper Functions ===
def get_features(form_data, disease_type):
    if disease_type not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown disease type: {disease_type}")

    features = []
    for feature in FEATURE_CONFIGS[disease_type]:
        if feature not in form_data:
            raise ValueError(f"Missing feature: {feature}")
        try:
            features.append(float(form_data[feature]))
        except ValueError:
            raise ValueError(f"Invalid input for {feature}: {form_data[feature]}")
    return features

def make_prediction(model, features):
    prediction = model.predict(features)[0]
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(features)[0].max() * 100
    else:
        confidence = 85 if prediction == 1 else 15
    return prediction, confidence

def get_risk_level(probability):
    if probability < 30:
        return {'level': 'Low Risk', 'color': 'success'}
    elif probability < 70:
        return {'level': 'Moderate Risk', 'color': 'warning'}
    else:
        return {'level': 'High Risk', 'color': 'danger'}

def get_recommendations(disease_type, prediction):
    if prediction == 1:
        return {
            'diabetes': ["Consult endocrinologist", "Monitor blood sugar", "Balanced diet", "Regular exercise"],
            'heart': ["Consult cardiologist", "Monitor blood pressure", "Low-sodium diet", "Stress management"],
            'parkinsons': ["Consult neurologist", "Physical therapy", "Occupational therapy"],
            'lung': ["Pulmonologist visit", "Quit smoking", "Avoid pollution"],
            'thyroid': ["Endocrinologist follow-up", "Regular thyroid checks"]
        }.get(disease_type, ["Consult a healthcare provider"])
    else:
        return {
            'diabetes': ["Exercise regularly", "Healthy diet", "Annual screening"],
            'heart': ["Regular check-ups", "Heart-healthy lifestyle"],
            'parkinsons': ["Monitor symptoms", "Stay active"],
            'lung': ["Avoid smoking", "Breathing exercises"],
            'thyroid': ["Routine thyroid checks", "Balanced iodine intake"]
        }.get(disease_type, ["Maintain a healthy lifestyle"])

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
