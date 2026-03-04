import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ==========================
# 1️⃣ Load Dataset
# ==========================
data = pd.read_csv("data.csv")

# Encode categorical columns
le = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])

X = data.drop("isFraud", axis=1)
y = data["isFraud"]

# ==========================
# 2️⃣ Feature Scaling
# ==========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================
# 3️⃣ Handle Imbalance
# ==========================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ==========================
# 4️⃣ Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.2,
    random_state=42
)

# ==========================
# 5️⃣ Train Model
# ==========================
# 🔹 Reduce dataset size BEFORE preprocessing
data = data.sample(150000, random_state=42)

X = data.drop("isFraud", axis=1)
y = data["isFraud"]

# 🔹 Skip SMOTE (very heavy)
# Instead use class_weight

model = RandomForestClassifier(
    n_estimators=25,        # small number of trees
    max_depth=10,           # limit tree depth
    class_weight="balanced",
    random_state=42,
    n_jobs=-1               # use all CPU cores
)

model.fit(X_train, y_train)

# ==========================
# 6️⃣ Predictions
# ==========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================
# 7️⃣ Evaluation Metrics
# ==========================

print("\n========== MODEL EVALUATION ==========\n")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

specificity = TN / (TN + FP)
sensitivity = recall  # same as recall

print(f"Accuracy           : {accuracy}")
print(f"Precision          : {precision}")
print(f"Recall (Sensitivity): {sensitivity}")
print(f"Specificity        : {specificity}")
print(f"F1 Score           : {f1}")
print(f"ROC-AUC Score      : {roc_auc}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# 8️⃣ Confusion Matrix Plot
# ==========================
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================
# 9️⃣ ROC Curve Plot
# ==========================
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()