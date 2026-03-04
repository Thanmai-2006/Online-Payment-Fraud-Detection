# ===============================
# TRAINING FILE (ERROR-FREE)
# ===============================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===============================
# STEP 1: Load Dataset
# ===============================

df = pd.read_csv("data.csv")  # make sure dataset name is correct

print("Dataset Loaded Successfully")

# ===============================
# STEP 2: Preprocessing
# ===============================

# Drop unnecessary columns
if "nameOrig" in df.columns:
    df = df.drop(["nameOrig"], axis=1)

if "nameDest" in df.columns:
    df = df.drop(["nameDest"], axis=1)

# Encode transaction type
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

# Separate features & target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# ===============================
# STEP 3: Scaling
# ===============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# STEP 4: Train Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# STEP 5: Train Model
# ===============================

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# STEP 6: Evaluate
# ===============================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# ===============================
# STEP 7: Save Everything
# ===============================

# Create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("Model, Scaler, and Label Encoder saved successfully!")

