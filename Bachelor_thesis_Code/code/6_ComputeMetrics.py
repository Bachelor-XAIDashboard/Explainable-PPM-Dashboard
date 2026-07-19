#helper script to compute the model metrics for the Dashboard

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,log_loss)

X = pd.read_csv("data/BPIC12_X_with_caseid.csv")
y = pd.read_csv("data/BPIC12_y.csv").squeeze()
test_case_ids = np.load("models/test_case_ids.npy")
# Split using same cases
is_test = X["case_id"].isin(test_case_ids)
X_test = X[is_test].drop(columns=["case_id"])
y_test = y[is_test]

rf_model = joblib.load("models/2_random_forest_bpic12.pkl")
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_bpic12.json")
label_encoder = joblib.load("models/label_encoder_bpic12.pkl")
# Encode labels for XGBoost
y_test_encoded = label_encoder.transform(y_test)

def top_k_accuracy(y_true, prob_matrix, k=3):
    top_k = np.argsort(prob_matrix, axis=1)[:, -k:]
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] in top_k[i]:
            correct += 1
    return correct / len(y_true)

#RF
rf_probs = rf_model.predict_proba(X_test)
rf_pred = np.argmax(rf_probs, axis=1)
rf_metrics = {
    "accuracy": float(accuracy_score(y_test_encoded, rf_pred)),
    "precision_macro": float(precision_score(y_test_encoded, rf_pred, average="macro")),
    "recall_macro": float(recall_score(y_test_encoded, rf_pred, average="macro")),
    "f1_macro": float(f1_score(y_test_encoded, rf_pred, average="macro")),
    "log_loss": float(log_loss(y_test_encoded, rf_probs)),
    "top3_accuracy": float(top_k_accuracy(y_test_encoded, rf_probs, k=3))
}
#XGBoost
dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
xgb_probs = xgb_model.predict(dtest)
xgb_pred = np.argmax(xgb_probs, axis=1)
xgb_metrics = {
    "accuracy": float(accuracy_score(y_test_encoded, xgb_pred)),
    "precision_macro": float(precision_score(y_test_encoded, xgb_pred, average="macro")),
    "recall_macro": float(recall_score(y_test_encoded, xgb_pred, average="macro")),
    "f1_macro": float(f1_score(y_test_encoded, xgb_pred, average="macro")),
    "log_loss": float(log_loss(y_test_encoded, xgb_probs)),
    "top3_accuracy": float(top_k_accuracy(y_test_encoded, xgb_probs, k=3))
}

with open("models/rf_metrics.json", "w") as f:
    json.dump(rf_metrics, f, indent=4)
with open("models/xgb_metrics.json", "w") as f:
    json.dump(xgb_metrics, f, indent=4)
