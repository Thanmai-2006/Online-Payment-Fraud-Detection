import joblib
import numpy as np

# Load model, scaler, and label encoder
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")  # make sure you saved this

print("\n=== Online Payment Fraud Prediction ===\n")

# Take inputs one by one
step = float(input("Enter step: "))
trans_type = input("Enter transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER): ").upper()
amount = float(input("Enter amount: "))
oldbalanceOrg = float(input("Enter old balance origin: "))
newbalanceOrig = float(input("Enter new balance origin: "))
oldbalanceDest = float(input("Enter old balance destination: "))
newbalanceDest = float(input("Enter new balance destination: "))

# Encode transaction type
try:
    type_encoded = label_encoder.transform([trans_type])[0]
except:
    print("Invalid transaction type entered!")
    exit()

# Create feature array
features = np.array([[step, type_encoded, amount,
                      oldbalanceOrg, newbalanceOrig,
                      oldbalanceDest, newbalanceDest]])

# Scale features
features = scaler.transform(features)

# Predict
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

print("\n==============================")
if prediction == 1:
    print(f"⚠️ Fraud Transaction Detected!")
else:
    print(f"✅ Safe Transaction")

print(f"Fraud Probability: {probability:.2f}")
print("==============================")
