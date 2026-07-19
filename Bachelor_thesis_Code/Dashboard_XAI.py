# Dashboard script

import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np
import json
import shap



# Loaders
@st.cache_data
def load_event_log():
    df = pd.read_csv("data/BPIC12_clean.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

@st.cache_data
def load_test_case_ids():
    return np.load("models/test_case_ids.npy")

@st.cache_resource
def load_models_and_features():
    features_df = pd.read_csv("data/BPIC12_X_with_caseid.csv")
    rf_model = joblib.load("models/2_random_forest_bpic12.pkl")
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/2xgboost_bpic12.json")
    label_encoder = joblib.load("models/label_encoder_bpic12.pkl")
    return features_df, rf_model, xgb_model, label_encoder

@st.cache_data
def load_metrics():
    with open("models/rf_metrics.json") as f:
        rf_metrics = json.load(f)
    with open("models/xgb_metrics.json") as f:
        xgb_metrics = json.load(f)
    return rf_metrics, xgb_metrics


# Helpers
def safe_inverse_transform(label_encoder, pred_encoded):
    classes = label_encoder.classes_
    try:
        pred_encoded_int = int(pred_encoded)
        if pred_encoded_int < 0 or pred_encoded_int >= len(classes):
            return "Unknown"
        return label_encoder.inverse_transform([pred_encoded_int])[0]
    except:
        return "Unknown"

def update_features(features, predicted_activity):
    features = features.copy()
    if "prefix_length" in features.columns:
        features["prefix_length"] += 1
    count_col = f"count_{predicted_activity}"
    if count_col in features.columns:
        features[count_col] += 1
    last_cols = [c for c in features.columns if c.startswith("last_activity_")]
    features[last_cols] = 0
    last_col = f"last_activity_{predicted_activity}"
    if last_col in features.columns:
        features[last_col] = 1
    return features

def predict_sequence_rf(model, features, steps, label_encoder):
    seq = []
    current = features.copy()
    for _ in range(steps):
        probs = model.predict_proba(current)[0]
        pred_idx = probs.argmax()
        pred_label = safe_inverse_transform(label_encoder, pred_idx)
        seq.append(pred_label)
        current = update_features(current, pred_label)
    return seq

def predict_sequence_xgb(model, features, steps, label_encoder):
    seq = []
    current = features.copy()
    for _ in range(steps):
        dmat = xgb.DMatrix(current, feature_names=current.columns)
        probs = model.predict(dmat)[0]
        pred_idx = probs.argmax()
        pred_label = safe_inverse_transform(label_encoder, pred_idx)
        seq.append(pred_label)
        current = update_features(current, pred_label)
    return seq

manager_actions = {
    "A_SUBMITTED":
        "A new loan application was submitted. "
        "Managers should ensure that the application enters the review process promptly.",
    "A_PARTLYSUBMITTED":
        "The application submission is incomplete. "
        "Additional customer information may still be required.",
    "A_PREACCEPTED":
        "The application was pre-accepted but still requires additional verification or supporting documents.",
    "A_ACCEPTED":
        "The application was accepted and is currently undergoing completeness screening.",
    "A_FINALIZED":
        "The application process has been finalized successfully. "
        "Managers may proceed with offer preparation or closing procedures.",
    "A_APPROVED":
        "The loan application has been approved successfully. "
        "Managers can proceed with customer onboarding and contract preparation.",
    "A_REGISTERED":
        "The approved application has been officially registered in the system.",
    "A_ACTIVATED":
        "The approved loan or service has been activated for the customer.",
    "A_DECLINED":
        "The application was declined. "
        "Managers may review rejection causes or communicate the decision to the customer.",
    "A_CANCELLED":
        "The application process was cancelled before completion.",
    "O_SELECTED":
        "The customer was selected to receive a loan offer. "
        "Managers should ensure the offer is prepared and communicated.",
    "O_CREATED":
        "A loan offer was created and is ready for preparation or transmission.",
    "O_SENT":
        "The offer was sent to the customer. "
        "Managers should monitor customer response and follow-up activities.",
    "O_SENT_BACK":
        "The customer returned the offer or requested modifications/additional clarification.",
    "O_ACCEPTED":
        "The customer accepted the offer. "
        "Managers may proceed with final approval and activation processes.",
    "O_DECLINED":
        "The customer declined the offer.",
    "O_CANCELLED":
        "The offer process was cancelled before completion.",
    "W_Afhandelen leads":
        "Employees are following up on incomplete or pending applications.",
    "W_Completeren aanvraag":
        "Additional application details are currently being completed.",
    "W_Nabellen offertes":
        "Employees are contacting customers regarding outstanding offers.",
    "W_Valideren aanvraag":
        "The application is currently being validated. "
        "Additional verification and risk assessment may be required.",
    "W_Nabellen incomplete dossiers":
        "Employees are requesting missing information or incomplete documentation.",
    "W_Beoordelen fraude":
        "The application is undergoing fraud assessment procedures.",
    "W_Wijzigen contractgegevens":
        "Approved contract details are currently being modified or updated."
}

activity_info = {
    "A_SUBMITTED":
        "Initial loan application submission by the customer.",
    "A_PARTLYSUBMITTED":
        "The application submission is incomplete.",
    "A_PREACCEPTED":
        "The application was pre-accepted but still requires additional information.",
    "A_ACCEPTED":
        "The application was accepted and is pending completeness screening.",
    "A_FINALIZED":
        "The application process was finalized successfully.",
    "A_APPROVED":
        "The loan application was approved successfully.",
    "A_REGISTERED":
        "The approved application was officially registered.",
    "A_ACTIVATED":
        "The approved loan or service was activated.",
    "A_DECLINED":
        "The loan application was declined.",
    "A_CANCELLED":
        "The application process was cancelled.",
    "O_SELECTED":
        "The applicant was selected to receive an offer.",
    "O_CREATED":
        "A loan offer was created.",
    "O_SENT":
        "The offer was sent to the customer.",
    "O_SENT_BACK":
        "The offer was returned for clarification or modification.",
    "O_ACCEPTED":
        "The customer accepted the offer.",
    "O_DECLINED":
        "The customer declined the offer.",
    "O_CANCELLED":
        "The offer process was cancelled.",
    "W_Afhandelen leads":
        "Following up on incomplete initial submissions.",
    "W_Completeren aanvraag":
        "Completing pre-accepted applications.",
    "W_Nabellen offertes":
        "Following up on submitted offers.",
    "W_Valideren aanvraag":
        "Validating and assessing the application.",
    "W_Nabellen incomplete dossiers":
        "Requesting missing information or documentation.",
    "W_Beoordelen fraude":
        "Investigating possible fraud cases.",
    "W_Wijzigen contractgegevens":
        "Modifying approved contract details."
}


st.set_page_config(layout="centered")
#st.set_page_config(layout="wide")
st.title("Explainable Decision Support Dashboard Using BPIC12")
st.info(
    "This dashboard combines predictive process monitoring "
    "with explainable AI to support business decision-making "
    "in financial process environments."
)

log_df = load_event_log()
test_case_ids = load_test_case_ids()
features_df, rf_model, xgb_model, label_encoder = load_models_and_features()
rf_metrics, xgb_metrics = load_metrics()
# SHAP explainer
rf_explainer = shap.TreeExplainer(rf_model)


with st.container():
    st.header("Event Log Overview")
    num_cases = log_df["case_id"].nunique()
    num_events = len(log_df)
    num_activities = log_df["activity"].nunique()
    case_lengths = log_df.groupby("case_id").size()
    avg_case_length = case_lengths.mean()
    case_duration = (
        log_df.groupby("case_id")["timestamp"].max()
        - log_df.groupby("case_id")["timestamp"].min()
    ).dt.total_seconds()
    avg_case_duration = case_duration.mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Cases", num_cases)
    col2.metric("Number of Events", num_events)
    col3.metric("Number of Activities", num_activities)
    col4, col5 = st.columns(2)
    col4.metric(
        "Average Case Length",
        f"{avg_case_length:.2f} events"
    )
    col5.metric(
        "Average Case Duration",
        f"{avg_case_duration / 3600:.2f} hours"
    )
    st.subheader("Activity Frequency")
    st.bar_chart(log_df["activity"].value_counts())


# Case selection
with st.container():
    st.header("Activity Reference Guide")
    selected_activity_info = st.selectbox(
        "Select an activity to view its meaning:",
        sorted(activity_info.keys())
    )
    st.info(activity_info[selected_activity_info])
    st.header("Decision Support Analysis")
    view_mode = st.radio(
        "Explanation Mode",
        ["Technical View", "Manager View"]
    )
    case_ids = log_df[
        log_df["case_id"].isin(test_case_ids)
    ]["case_id"].unique()
    case_id = st.selectbox(
        "Select a test case",
        case_ids
    )
    case_events = (
        log_df[log_df["case_id"] == case_id]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    st.subheader("Event Log for Selected Case")
    st.dataframe(case_events[["activity", "timestamp"]])
    num_events = len(case_events)
    if num_events < 2:
        st.warning("Case has only one event. No prediction possible.")
        st.stop()
    prefix_length = st.slider(
        "Select prefix length",
        min_value=1,
        max_value=num_events - 1,
        value=num_events - 1
    )
    correct_next_activity = case_events.loc[
        prefix_length,
        "activity"
    ]
    st.markdown(
        f"### Correct next activity: **{correct_next_activity}**"
    )

    
prefix_features = features_df[
    (features_df["case_id"] == case_id)
    &
    (features_df["prefix_length"] == prefix_length)
].drop(columns=["case_id"])

#RF
rf_probs = rf_model.predict_proba(prefix_features)[0]
rf_top_idx = rf_probs.argmax()
rf_top_label = safe_inverse_transform(
    label_encoder,
    rf_top_idx
)
rf_confidence = rf_probs[rf_top_idx]
rf_top3_idx = rf_probs.argsort()[-3:][::-1]
rf_top3 = [
    (
        safe_inverse_transform(label_encoder, i),
        rf_probs[i]
    )
    for i in rf_top3_idx
]

#SHAP
shap_values = rf_explainer.shap_values(prefix_features)
if isinstance(shap_values, list):
    rf_shap_vals = shap_values[rf_top_idx][0]
else:
    rf_shap_vals = shap_values[0, :, rf_top_idx]
shap_df = pd.DataFrame({
    "feature": prefix_features.columns,
    "impact": rf_shap_vals
})

shap_df["abs_impact"] = shap_df["impact"].abs()

shap_df = (
    shap_df
    .sort_values(by="abs_impact", ascending=False)
    .head(5)
)

# XGBoost
dmat = xgb.DMatrix(
    prefix_features,
    feature_names=prefix_features.columns
)
xgb_probs = xgb_model.predict(dmat)[0]
xgb_top_idx = xgb_probs.argmax()
xgb_top_label = safe_inverse_transform(
    label_encoder,
    xgb_top_idx
)
xgb_confidence = xgb_probs[xgb_top_idx]
xgb_top3_idx = xgb_probs.argsort()[-3:][::-1]
xgb_top3 = [
    (
        safe_inverse_transform(label_encoder, i),
        xgb_probs[i]
    )
    for i in xgb_top3_idx
]

# PREDICTION DISPLAY

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.subheader("Random Forest")
    st.write(
        f"Predicted Activity: **{rf_top_label}**"
    )
    # TECHNICAL VIEW
    if view_mode == "Technical View":
        st.write(
            f"Confidence: **{rf_confidence:.2f}**"
        )
        st.progress(float(rf_confidence))
        for label, prob in rf_top3:
            st.write(f"{label}: {prob:.3f}")
        st.divider()
        st.subheader("Why this prediction?")
        for _, row in shap_df.iterrows():
            feature = row["feature"]
            direction = (
                "increases"
                if row["impact"] > 0
                else "decreases"
            )
            if "count_" in feature:
                activity = feature.replace(
                    "count_",
                    ""
                )
                st.write(
                    f"Previous occurrences of "
                    f"activity '{activity}' "
                    f"{direction} the likelihood "
                    f"of this prediction."
                )
            elif "elapsed_time" in feature:
                st.write(
                    "The total process duration "
                    "strongly influences the prediction."
                )
            elif "time_since_prev" in feature:
                st.write(
                    "The time between process steps "
                    "affects the prediction."
                )
            elif "prefix_length" in feature:
                st.write(
                    "The current stage of the process "
                    "impacts the predicted outcome."
                )
            else:
                st.write(
                    f"{feature} influences the prediction."
                )
        #st.bar_chart(shap_df.set_index("feature")["impact"])
        st.subheader("SHAP Feature Impact")
        chart_df = shap_df.copy()
        chart_df = chart_df.sort_values(by="impact")
        st.bar_chart(
            chart_df.set_index("feature")["impact"]
        )
    # MANAGER VIEW
    else:
        st.caption(
            "Confidence thresholds: "
        "High ≥ 0.90 | Moderate ≥ 0.70 | Low < 0.70"
        )
        confidence_level = (
            "High"
            if rf_confidence >= 0.9
            else "Moderate"
            if rf_confidence >= 0.7
            else "Low"
        )
        st.write(
            f"Confidence Level: **{confidence_level}**"
        )
        st.divider()
        st.subheader("Decision Support Summary")
        st.write(
            f"The system predicts the next activity "
            f"'{rf_top_label}' with confidence "
            f"({rf_confidence:.2f})."
        )
        st.divider()
        st.subheader("Business Explanation")
        if rf_confidence >= 0.9:
            st.success(
                "The process appears to follow "
                "a stable and recognizable pattern."
            )
        elif rf_confidence >= 0.7:
            st.info(
                "The prediction confidence is moderate. "
                "The case should still be reviewed carefully."
            )
        else:
            st.warning(
                "The prediction confidence is low. "
                "Human review is strongly recommended."
            )
        shown_messages = set()
        for _, row in shap_df.iterrows():
            feature = row["feature"]
            if "count_" in feature:
                msg = (
                    "Previous process activities "
                    "contribute to the prediction."
                )
            elif "elapsed_time" in feature:
                msg = (
                    "The overall process duration "
                    "strongly influences the prediction."
                )
            elif "time_since_prev" in feature:
                msg = (
                    "The time between activities "
                    "affects the process outcome."
                )
            elif "prefix_length" in feature:
                msg = (
                    "The current stage of the process "
                    "impacts the prediction."
                )
            else:
                msg = (
                    "Additional process characteristics "
                    "influence the prediction."
                )
            if msg not in shown_messages:
                st.write(msg)
                shown_messages.add(msg)
        st.divider()
        st.subheader("Recommended Action")
        recommended_action = manager_actions.get(
            rf_top_label,
            "No managerial recommendation available."
        )
        st.info(recommended_action)
        #if "O_DECLINED" in rf_top_label:
            #st.error(
               # "Potential rejection risk detected. "
                #"Review the application and request "
                #"additional supporting information."
            #)
       #elif "A_ACCEPTED" in rf_top_label:
            #st.success(
                #"The process appears likely to continue "
                #"successfully. Standard processing can proceed."
            #)
        #elif "W_Valideren aanvraag" in rf_top_label:
            #st.info(
                #"Validation-related activity predicted. "
               # "The case may require additional verification steps."
            #)
        #else:
            #st.info(
              # "Monitor the process progression "
               # "and review the case if necessary."
           # )

        
# XGBOOST PANEL
with pred_col2:
    st.subheader("XGBoost")
    st.write(
        f"Predicted Activity: **{xgb_top_label}**"
    )
    if view_mode == "Technical View":
        st.write(
            f"Confidence: **{xgb_confidence:.2f}**"
        )
        st.progress(float(xgb_confidence))
        for label, prob in xgb_top3:
            st.write(f"{label}: {prob:.3f}")
    else:
        confidence_level = (
            "High"
            if xgb_confidence >= 0.9
            else "Moderate"
            if xgb_confidence >= 0.7
            else "Low"
        )
        st.write(
            f"Confidence Level: **{confidence_level}**"
        )


# Future sequence prediction
#st.header("Future Sequence Prediction")

#future_steps = st.slider(
#    "Number of future steps to predict",
#    min_value=1,
#    max_value=5,
#    value=3
#)

#rf_sequence = predict_sequence_rf(
#    rf_model,
#    prefix_features.copy(),
#    future_steps,
#    label_encoder
#)

#xgb_sequence = predict_sequence_xgb(
#    xgb_model,
#    prefix_features.copy(),
#    future_steps,
#    label_encoder
#)

#seq_col1, seq_col2 = st.columns(2)

#with seq_col1:
#    st.subheader("Random Forest Future")
#    for i, act in enumerate(rf_sequence, start=1):
#        st.write(f"{i}. {act}")

#with seq_col2:
#    st.subheader("XGBoost Future")
#    for i, act in enumerate(xgb_sequence, start=1):
#        st.write(f"{i}. {act}")


# Model eval
st.header("Model Performance (Test Set)")

met_col1, met_col2 = st.columns(2)

with met_col1:
    st.subheader("Random Forest")
    st.metric(
        "Accuracy",
        f"{rf_metrics['accuracy']:.3f}"
    )
    st.metric(
        "Precision (macro)",
        f"{rf_metrics['precision_macro']:.3f}"
    )
    st.metric(
        "Recall (macro)",
        f"{rf_metrics['recall_macro']:.3f}"
    )
    st.metric(
        "F1-score (macro)",
        f"{rf_metrics['f1_macro']:.3f}"
    )
    st.metric(
        "Log Loss",
        f"{rf_metrics['log_loss']:.3f}"
    )
    st.metric(
        "Top-3 Accuracy",
        f"{rf_metrics['top3_accuracy']:.3f}"
    )

with met_col2:
    st.subheader("XGBoost")
    st.metric(
        "Accuracy",
        f"{xgb_metrics['accuracy']:.3f}"
    )
    st.metric(
        "Precision (macro)",
        f"{xgb_metrics['precision_macro']:.3f}"
    )
    st.metric(
        "Recall (macro)",
        f"{xgb_metrics['recall_macro']:.3f}"
    )
    st.metric(
        "F1-score (macro)",
        f"{xgb_metrics['f1_macro']:.3f}"
    )
    st.metric(
        "Log Loss",
        f"{xgb_metrics['log_loss']:.3f}"
    )
    st.metric(
        "Top-3 Accuracy",
        f"{xgb_metrics['top3_accuracy']:.3f}"
    )