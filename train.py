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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# AdaBoost
ab_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ab_model.fit(X_train, y_train)

# ===============================
# STEP 6: Evaluate
# ===============================

# Random Forest Accuracy
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Logistic Regression Accuracy
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Decision Tree Accuracy
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# AdaBoost Accuracy
ab_pred = ab_model.predict(X_test)
ab_acc = accuracy_score(y_test, ab_pred)

print("Random Forest Accuracy:", rf_acc)
print("Logistic Regression Accuracy:", lr_acc)
print("Decision Tree Accuracy:", dt_acc)
print("AdaBoost Accuracy:", ab_acc)

# ===============================
# STEP 7: Save Everything
# ===============================

joblib.dump(rf_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")

