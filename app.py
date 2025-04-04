from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import urllib.parse
import tldextract
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained models
rf_model = joblib.load("random_forest_model.pkl")
svm_model = joblib.load("svm_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")  # Load the trained scaler

# Function to extract only 8 features from a URL
def extract_features(url):
    parsed_url = urllib.parse.urlparse(url)
    domain_info = tldextract.extract(url)
    
    try:
        features = [
            len(parsed_url.path),  # directory_length
            1 if "https" in parsed_url.scheme else 0,  # time_domain_activation
            len(url),  # length_url
            parsed_url.path.count('/'),  # qty_slash_directory
            len(domain_info.domain),  # ttl_hostname
            url.count('.'),  # qty_dot_file
            sum(c.isdigit() for c in domain_info.domain),  # asn_ip (checking digits in domain)
            len(parsed_url.netloc)  # time_response
        ]
        print(f"Extracted Features: {features}")  # Debugging Output
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        url = data.get("url", "").strip()

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        features = extract_features(url)

        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        # Scale input features
        features_scaled = scaler.transform([features])

        # Predict with models
        rf_prediction = rf_model.predict(features_scaled)[0]
        svm_prediction = svm_model.predict(features_scaled)[0]
        xgb_prediction = xgb_model.predict(features_scaled)[0]
        lr_prediction = lr_model.predict(features_scaled)[0]

        # Hybrid model - majority voting
        predictions = [rf_prediction, svm_prediction, xgb_prediction, lr_prediction]
        hybrid_prediction = 1 if sum(predictions) > 2 else 0  # Majority voting

        result = "Phishing" if hybrid_prediction == 1 else "Legitimate"

        response = {
            "rf_prediction": "Phishing" if rf_prediction == 1 else "Legitimate",
            "svm_prediction": "Phishing" if svm_prediction == 1 else "Legitimate",
            "xgb_prediction": "Phishing" if xgb_prediction == 1 else "Legitimate",
            "lr_prediction": "Phishing" if lr_prediction == 1 else "Legitimate",
            "hybrid_prediction": result
        }
        
        print(f"Predictions: {response}")  # Debugging Output
        return jsonify(response)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
