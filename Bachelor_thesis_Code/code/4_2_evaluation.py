# Evaluate Random Forest and XGBoost models using case-based test split

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load data
X_full = pd.read_csv("data/BPIC12_X_with_caseid.csv")
y = pd.read_csv("data/BPIC12_y.csv").squeeze()
test_case_ids = np.load("models/test_case_ids.npy")

# Split using case IDs
is_test = X_full["case_id"].isin(test_case_ids)
X_test = X_full[is_test].drop(columns=["case_id"])
y_test = y[is_test]
X_train = X_full[~is_test].drop(columns=["case_id"])
y_train = y[~is_test]

# Load models
rf_model = joblib.load("models/2_random_forest_bpic12.pkl")
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_bpic12.json")
label_encoder = joblib.load("models/label_encoder_bpic12.pkl")

# Evaluate Random Forest
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_report = classification_report(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)

# Save results
with open("random_forest_eval.txt", "w", encoding="utf-8") as f:
    f.write("Random Forest Evaluation\n")
    f.write(f"Accuracy: {rf_accuracy:.4f}\n\n")
    f.write(rf_report)

# evaluate XGBoost
# Encode labels for XGBoost comparison
y_test_encoded = label_encoder.transform(y_test)
# Create DMatrix
dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())
# Predict probabilities
y_prob = xgb_model.predict(dtest)
# Convert to class index
y_pred_encoded = np.argmax(y_prob, axis=1)
# Convert back to activity labels
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
xgb_accuracy = accuracy_score(y_test, y_pred_labels)
xgb_report = classification_report(y_test, y_pred_labels)
xgb_cm = confusion_matrix(y_test, y_pred_labels)

# Save results
with open("xgboost_eval.txt", "w", encoding="utf-8") as f:
    f.write("XGBoost Evaluation\n")
    f.write(f"Accuracy: {xgb_accuracy:.4f}\n\n")
    f.write(xgb_report)

# Feature Importance (Random Forest)
#import matplotlib.pyplot as plt

#def plot_feature_importance(model, feature_names, title="Feature Importance", top_n=20):
#    if hasattr(model, "feature_importances_"):
#        importances = model.feature_importances_
#        indices = np.argsort(importances)[::-1][:top_n]

#        plt.figure(figsize=(10, 6))
#        plt.title(title)
#        plt.bar(range(top_n), importances[indices])
#        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=90)
#        plt.tight_layout()
#        plt.show()
#    else:
#        print(f"{title}: Model has no feature_importances_ attribute.")

#plot_feature_importance(rf_model, X_test.columns, "Random Forest Feature Importance")
