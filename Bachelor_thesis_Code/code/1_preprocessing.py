# 1_preprocessing.py
# Purpose: Load BPIC12 event log (as DataFrame), clean, sort, and prepare for feature encoding

import pm4py
import pandas as pd

# load BPIC12
file_path = "BPI_Challenge_2012.xes"
log = pm4py.read_xes(file_path)

# keep only relevant columns
# For next activity prediction, we only need case ID, activity, and timestamp
df = log[['case:concept:name', 'concept:name', 'time:timestamp']].copy()

# rename columns for clarity
df = df.rename(columns={'case:concept:name': 'case_id', 'concept:name': 'activity', 'time:timestamp': 'timestamp'})

# convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

#sort events by case_id and timestamp
df = df.sort_values(by=['case_id', 'timestamp']).reset_index(drop=True)

# Remove cases with fewer than 2 events
case_counts = df['case_id'].value_counts()
valid_cases = case_counts[case_counts >= 2].index

df = df[df['case_id'].isin(valid_cases)].reset_index(drop=True)
#Save
df.to_csv("BPIC12_clean.csv", index=False)



