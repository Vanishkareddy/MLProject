import pandas as pd
import numpy as np
import joblib
import re
import tldextract
import urllib.parse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("dataset_small.csv", encoding="latin1")

# Select only 8 important features
selected_features = ["directory_length", "time_domain_activation", "length_url", "qty_slash_directory",
                     "ttl_hostname", "qty_dot_file", "asn_ip", "time_response"]
X = df[selected_features]
y = df['phishing']

# Standardize selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Model Predictions
rf_pred = rf_model.predict(X_test)
svm_pred = svm_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Hybrid model (Average of all predictions)
final_pred = np.round((rf_pred + svm_pred + lr_pred + xgb_pred) / 4).astype(int)

# Accuracy Scores
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)
lr_acc = accuracy_score(y_test, lr_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)
final_acc = accuracy_score(y_test, final_pred)

# Print accuracy results
print(f"🔹 Random Forest Accuracy: {rf_acc:.4f}")
print(f"🔹 SVM Accuracy: {svm_acc:.4f}")
print(f"🔹 Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"🔹 XGBoost Accuracy: {xgb_acc:.4f}")
print(f"🔹 Hybrid Model Accuracy: {final_acc:.4f}")

# Save Models
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(lr_model, "logistic_regression_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
