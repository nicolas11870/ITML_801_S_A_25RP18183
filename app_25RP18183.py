from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import numpy as np

app = Flask(__name__, template_folder="templates")
CORS(app)

# =====================================================
# PATHS (CORRECTED FOR NEW PROJECT STRUCTURE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEPLOYMENT_DIR = os.path.join(BASE_DIR, "deployment")

MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "heart_disease_model_25RP18183.pkl")
FEATURES_PATH = os.path.join(DEPLOYMENT_DIR, "feature_columns.txt")
CLASSES_PATH = os.path.join(DEPLOYMENT_DIR, "class_names.txt")

# =====================================================
# LOAD MODEL & ARTIFACTS
# =====================================================
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open(FEATURES_PATH, "r") as f:
        feature_columns = [line.strip() for line in f.readlines()]
    print(f"{len(feature_columns)} feature columns loaded")
except Exception as e:
    print(f"Error loading feature columns: {e}")
    feature_columns = []

try:
    with open(CLASSES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Class names loaded: {class_names}")
except Exception as e:
    print(f"Error loading class names: {e}")
    class_names = ["No Disease", "Very Mild", "Mild", "Severe", "Immediate Danger"]

# =====================================================
# VALUE MAPPINGS (FOR READABLE LOGGING)
# =====================================================
value_map = {
    "sex": {0: "Female", 1: "Male"},
    "fbs": {0: "No", 1: "Yes"},
    "exang": {0: "No", 1: "Yes"},
    "cp": {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-Anginal Pain",
        3: "Asymptomatic"
    },
    "restecg": {
        0: "Normal",
        1: "ST-T Abnormality",
        2: "LV Hypertrophy"
    },
    "slope": {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping"
    },
    "thal": {
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    },
    "ca": {
        0: "0 vessels",
        1: "1 vessel",
        2: "2 vessels",
        3: "3 vessels"
    }
}

# =====================================================
# OPTIONAL TERMINAL TESTING
# =====================================================
def test_patient_predictions():
    if model is None or not feature_columns:
        print("Model or feature columns not loaded. Skipping test predictions.")
        return

    print("\n=== TEST PATIENT PREDICTIONS ===")
    sample_patients = [
        {"age": 55, "sex": 1, "cp": 0, "trestbps": 140, "chol": 250, "fbs": 0,
         "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 2},

        {"age": 60, "sex": 0, "cp": 2, "trestbps": 130, "chol": 210, "fbs": 1,
         "restecg": 0, "thalach": 160, "exang": 1, "oldpeak": 2.5, "slope": 1, "ca": 1, "thal": 3}
    ]

    for patient in sample_patients:
        df = pd.DataFrame([patient], columns=feature_columns)
        probs = model.predict_proba(df)[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        # Print patient input
        print("\nPatient Input:")
        for k, v in patient.items():
            if k in value_map:
                print(f"{k}: {value_map[k].get(v, v)}")
            else:
                print(f"{k}: {v}")

        # Print predicted class with confidence
        print(f"Predicted Class: {class_names[pred_idx]} ({confidence*100:.1f}%)")

        # Print all class probabilities as percentages
        class_percentages = [f"{class_names[i]}: {probs[i]*100:.1f}%" for i in range(len(class_names))]
        print("Class Probabilities:")
        print("  " + ", ".join(class_percentages))

        print("-" * 50)

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("index_25RP18183.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": len(feature_columns) == 13,
        "classes_loaded": len(class_names) == 5
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No input data provided"}), 400

    input_vec = {}
    for col in feature_columns:
        if col not in data:
            return jsonify({"status": "error", "message": f"Missing feature: {col}"}), 400
        try:
            input_vec[col] = float(data[col])
        except ValueError:
            return jsonify({"status": "error", "message": f"Invalid value for {col}"}), 400

    df = pd.DataFrame([input_vec], columns=feature_columns)
    probs = model.predict_proba(df)[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    risk_map = {
        0: {"level": "No Disease", "color": "#28a745"},
        1: {"level": "Very Mild", "color": "#17a2b8"},
        2: {"level": "Mild", "color": "#ffc107"},
        3: {"level": "Severe", "color": "#fd7e14"},
        4: {"level": "Immediate Danger", "color": "#dc3545"}
    }

    response = {
        "status": "success",
        "prediction": {
            "class": pred_idx,
            "class_name": class_names[pred_idx],
            "confidence": confidence,
            "risk_level": risk_map[pred_idx]["level"],
            "risk_color": risk_map[pred_idx]["color"]
        },
        "probabilities": [
            {
                "class": i,
                "class_name": class_names[i],
                "probability": float(probs[i])
            } for i in range(len(class_names))
        ]
    }

    return jsonify(response)

# =====================================================
# RUN APPLICATION
# =====================================================
if __name__ == "__main__":
    print("=== HEART DISEASE RISK PREDICTION API RUNNING ===")
    test_patient_predictions()
    app.run(host="0.0.0.0", port=5000, debug=True)
