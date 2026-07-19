# train random forest with same training set as XGBoost

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

#Load 
X = pd.read_csv("BPIC12_X_with_caseid.csv")  #includes case_id
y = pd.read_csv("BPIC12_y.csv").squeeze()
test_case_ids = np.load("models/test_case_ids.npy")  # IDs from XGBoost script

# Split into train/test using test_case_ids
is_test = X["case_id"].isin(test_case_ids)
X_test = X[is_test].drop(columns=["case_id"])
y_test = y[is_test]

X_train = X[~is_test].drop(columns=["case_id"])
y_train = y[~is_test]

# Train
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}\n")
print(f"Classification report: {classification_report(y_test, y_pred)}\n")
print(f"First 10 classes: {confusion_matrix(y_test, y_pred)[:10, :10]}\n")

# Save model
joblib.dump(rf, "2_random_forest_bpic12.pkl")

