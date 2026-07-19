#XGBoost training script

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import numpy as np
from collections import Counter


# Load features and target (with case_id)
X_full = pd.read_csv("BPIC12_X_with_caseid.csv")
y = pd.read_csv("BPIC12_y_with_caseid.csv").squeeze()

# Split by case
case_ids = X_full['case_id'].unique()
np.random.seed(42)
test_cases = np.random.choice(case_ids, size=int(0.2*len(case_ids)), replace=False)
train_cases = np.setdiff1d(case_ids, test_cases)
train_mask = X_full['case_id'].isin(train_cases)
test_mask = X_full['case_id'].isin(test_cases)

# Drop case_id for model training
X_train = X_full[train_mask].drop(columns=['case_id'])
y_train = y[train_mask]
X_test = X_full[test_mask].drop(columns=['case_id'])
y_test = y[test_mask]

# Save test case IDs for the dashboard
np.save("models/test_case_ids.npy", test_cases)

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
num_classes = len(le.classes_)

# Handle class imbalance
class_counts = Counter(y_train_encoded)
total_count = len(y_train_encoded)
sample_weights = [total_count / class_counts[c] for c in y_train_encoded]

# create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train_encoded, weight=sample_weights)
dtest = xgb.DMatrix(X_test, label=y_test_encoded)

#set parameters
params = {
    'objective': 'multi:softprob',  # get probabilities
    'num_class': num_classes,
    'max_depth': 5,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'mlogloss',
    'seed': 42
}

#Train
num_rounds = 100
bst = xgb.train(params, dtrain, num_boost_round=num_rounds)

#Predict & evaluate
y_prob = bst.predict(dtest)           # shape: (n_samples, n_classes)
y_pred = y_prob.argmax(axis=1)        # pick most likely class
y_pred_labels = le.inverse_transform(y_pred) #convert back into activity names
y_test_labels = le.inverse_transform(y_test_encoded) #also convert back to activity names

# Save model & encoder
bst.save_model("models/xgboost_bpic12.json")
joblib.dump(le, "models/label_encoder_bpic12.pkl")

