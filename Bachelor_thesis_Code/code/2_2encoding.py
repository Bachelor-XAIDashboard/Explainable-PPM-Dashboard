# 2_encoding_with_caseid.py
# Purpose: Generate prefix-based features from cleaned BPIC12 DataFrame
# Input: df with columns ['case_id', 'activity', 'timestamp']
# Output: X (features with case_id), y (next activity)

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from dateutil import parser

# load DataFrame
df = pd.read_csv("BPIC12_clean.csv")
# Clean whitespace and quotes
df['timestamp'] = df['timestamp'].str.strip().str.replace('"', '')
# Convert timestamps safely using dateutil parser
df['timestamp'] = df['timestamp'].apply(lambda x: parser.isoparse(x))

# initialize lists for features and targets
feature_rows = []
target_rows = []

# Iterate over each case and generate prefixes
case_ids = df['case_id'].unique()

for case_id in case_ids:
    case_events = df[df['case_id'] == case_id].sort_values('timestamp')
    activities = case_events['activity'].tolist()
    timestamps = case_events['timestamp'].tolist()
    # Generate prefixes of length 1 to n-1
    for i in range(1, len(activities)):
        prefix = activities[:i]
        next_activity = activities[i]  # target
        # Features
        feature_dict = {}
        feature_dict['case_id'] = case_id          # include case_id
        feature_dict['prefix_length'] = i          # prefix length option
        # Last activity (categorical)
        feature_dict['last_activity'] = prefix[-1]
        # Count of each activity in prefix
        counts = pd.Series(prefix).value_counts().to_dict()
        for act, count in counts.items():
            feature_dict[f'count_{act}'] = count
        # Time features
        # Time since previous event in seconds
        time_diff = (timestamps[i-1] - timestamps[i-2]).total_seconds() if i > 1 else 0
        feature_dict['time_since_prev'] = time_diff
        # Total elapsed time since case start
        feature_dict['elapsed_time'] = (timestamps[i-1] - timestamps[0]).total_seconds()
        feature_rows.append(feature_dict)
        target_rows.append(next_activity)

#Convert to DataFrame
X = pd.DataFrame(feature_rows)
y = pd.Series(target_rows, name='next_activity')


# Handle categorical features (last_activity)
categorical_cols = ['last_activity']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cat = encoder.fit_transform(X[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))
X = X.drop(columns=categorical_cols).reset_index(drop=True)
X = pd.concat([X, encoded_cat_df], axis=1)

# Save for model training
X.to_csv("BPIC12_X_with_caseid.csv", index=False)
y.to_csv("BPIC12_y_with_caseid.csv", index=False)
