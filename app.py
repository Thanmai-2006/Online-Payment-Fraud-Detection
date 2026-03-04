from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load saved files
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:
        step = float(request.form["step"])
        trans_type = request.form["type"].upper()
        amount = float(request.form["amount"])
        oldbalanceOrg = float(request.form["oldbalanceOrg"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        # Encode type
        type_encoded = label_encoder.transform([trans_type])[0]

        # Create feature array
        features = np.array([[step, type_encoded, amount,
                              oldbalanceOrg, newbalanceOrig,
                              oldbalanceDest, newbalanceDest]])

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        if prediction == 1:
            result = f"⚠ Fraud Transaction (Risk: {probability:.2f})"
        else:
            result = f"✅ Safe Transaction (Risk: {probability:.2f})"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Invalid Input!")

if __name__ == "__main__":
    app.run(debug=True)
