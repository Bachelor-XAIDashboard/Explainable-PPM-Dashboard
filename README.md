# Explainable Decision Support Dashboard for Predictive Process Monitoring

An Explainable Artificial Intelligence (XAI) dashboard developed as part of my Bachelor's Thesis in Business Informatics at Saarland University.

The dashboard combines machine learning-based Predictive Process Monitoring with Explainable AI techniques to improve transparency and support managerial decision-making.

---

## Overview

Traditional Predictive Process Monitoring systems often rely on high-performing machine learning models whose internal decision-making remains difficult to understand.

This project addresses this challenge by integrating Explainable AI into a decision support dashboard that provides both:

- **Technical explanations** for data scientists and technical users
- **Managerial explanations** for business users and decision-makers

The dashboard allows users to understand not only the predicted next process activity but also the reasoning behind the prediction and its business implications.

---

## Features

### Predictive Process Monitoring

- Next Activity Prediction
- Random Forest ML model
- XGBoost ML model
- Prediction confidence scores
- Top-3 prediction probabilities

### Explainable AI

- SHAP feature importance
- Feature impact visualisation
- Technical explanation layer

### Managerial Decision Support

- Business-oriented explanations
- Decision Support Summary
- Recommended managerial actions
- Activity reference guide
- Confidence categorisation (High / Moderate / Low)

---

## Dashboard Components

The dashboard consists of several integrated components:

- Event Log Overview
- Activity Reference Guide
- Case Selection
- Prefix Selection
- Technical Explainability View
- SHAP Feature-based Explanations
- SHAP Feature Impact Plot
- Managerial Explainability View
- Decision Support Summary
- Business Explanation
- Recommended Actions

---

## Tools Used

- Python
- Streamlit
- Pandas
- Pm4py
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- dateutil
- joblib
- xgbosst
- json

---

## Dataset

This project uses the publicly available **BPIC12 Event Log**, which represents a real-world loan application process from a Dutch financial institution.

Dataset:
https://data.4tu.nl/articles/dataset/BPI_Challenge_2012/12689204

---

## Repository Structure

```
Bachelor_thesis_Code/
│
├── code/
│   ├── 1_preprocessing.py
│   ├── 2_encoding.py
│   ├── 3_Random_Forest.py
│   ├── 3_Gradient_Boost.py
│   ├── 4_2_evaluation.py
│   └── 6_ComputeMetrics.py
│
└── Dashboard_XAI.py

Evaluation of an Explainable AI Dashboard for Predictive Process Monitoring.xlsx
```

---

## Running the Dashboard

Install the required Python packages.
Run the scripts from the "code" folder in order, staring from 1_preprocessing.py to 6_ComputeMetrics.

Then launch the dashboard with:

```bash
streamlit run Dashboard_XAI.py
```

---

## Research Context

This repository accompanies my Bachelor's Thesis:

**"Explainable AI in Predictive Process Monitoring: Development and Evaluation of an Explainable Decision Support Dashboard for Managerial Decision-Making"**

The developed dashboard was evaluated using six Human-Centered Explainable AI constructs:

- Trust
- Comprehensibility
- Actionability
- Usability
- Cognitive Load
- Simulatability

The anonymized survey responses and the complete evaluation results are available in:
**`Evaluation of the Explainable AI Dashboard for Predictive Process Monitoring.xlsx`**

### Survey Results

A summary of the evaluation results can be found here:

**Survey Results:** *https://forms.office.com/Pages/AnalysisPage.aspx?AnalyzerToken=a4HPG8biNzFAz49lI7KF1l5vvuA6C3HS&id=DQSIkWdsW0yxEjajBLZtrQAAAAAAAAAAAAO__QSYPFRUQlVYMzNXRFg1OEtRMUozOTEzSzNRSVdUVS4u*

---

## Future Improvements

Potential extensions include:

- SHAP explanations for XGBoost
- More detailed feature explanations ("how" features influence predictions)
- Risk and urgency indicators
- More specific recommended actions
- Multi-domain evaluation
- Larger user studies

---

## Author

**Abderrahmane Chakroune**

Bachelor of Science in Business Informatics

Saarland University
