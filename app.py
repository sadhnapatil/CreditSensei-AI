import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# Load Model
model = pickle.load(open("credit_model.pkl", "rb"))

# Page Configuration
st.set_page_config(
    page_title="CreditSensei AI",
    page_icon="💳",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.stApp{
    background-color:#0f172a;
}

.main-title{
    text-align:center;
    color:white;
    font-size:45px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("logo.png", width=250)
st.sidebar.title("CreditSensei AI")

page = st.sidebar.radio(
    "Navigation",
    ["Credit Assessment", "About"]
)

# ====================================================
# CREDIT ASSESSMENT PAGE
# ====================================================

if page == "Credit Assessment":

    st.markdown(
        '<p class="main-title">💳 CreditSensei AI</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">AI Powered Credit Risk Prediction System</p>',
        unsafe_allow_html=True
    )

    st.info("""
Welcome to CreditSensei AI.

This intelligent credit assessment platform helps
financial institutions evaluate loan applications
using Machine Learning.
""")

    # Dashboard Metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🤖 Model", "Random Forest")

    with c2:
        st.metric("🎯 Accuracy", "78%")

    with c3:
        st.metric("📊 Features", "10")

    with c4:
        st.metric("💾 Dataset", "German Credit")

    st.divider()

    st.subheader("📈 Model Performance")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric("Precision", "80.0%")

    with p2:
        st.metric("Recall", "91.4%")

    with p3:
        st.metric("F1 Score", "85.3%")

    with p4:
        st.metric("ROC-AUC", "77.0%")

    st.divider()

    # Customer Form
    with st.container(border=True):

        st.subheader("📝 Customer Credit Profile")

        col1, col2 = st.columns(2)

        with col1:

            age = st.slider(
                "Age",
                18,
                75,
                25
            )

            sex = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            job = st.selectbox(
                "Job Level",
                [0, 1, 2, 3]
            )

            housing = st.selectbox(
                "Housing",
                ["own", "rent", "free"]
            )

        with col2:

            saving = st.selectbox(
                "Saving Account",
                ["little", "moderate", "quite rich", "rich"]
            )

            checking = st.selectbox(
                "Checking Account",
                ["little", "moderate", "rich"]
            )

            credit_amount = st.number_input(
                "Credit Amount",
                min_value=100
            )

            duration = st.number_input(
                "Loan Duration (Months)",
                min_value=1
            )

        purpose = st.selectbox(
            "Purpose",
            [
                "radio/TV",
                "education",
                "furniture/equipment",
                "car",
                "business",
                "domestic appliances",
                "repairs",
                "vacation/others"
            ]
        )

    # Encoding Maps
    sex_map = {
        "Male": 1,
        "Female": 0
    }

    housing_map = {
        "own": 1,
        "rent": 2,
        "free": 0
    }

    saving_map = {
        "little": 0,
        "moderate": 1,
        "quite rich": 2,
        "rich": 3
    }

    checking_map = {
        "little": 0,
        "moderate": 1,
        "rich": 2
    }

    purpose_map = {
        "radio/TV": 0,
        "education": 1,
        "furniture/equipment": 2,
        "car": 3,
        "business": 4,
        "domestic appliances": 5,
        "repairs": 6,
        "vacation/others": 7
    }

    st.divider()

    if st.button("🚀 Predict Credit Risk", use_container_width=True):

        data = np.array([
            age,
            sex_map[sex],
            job,
            housing_map[housing],
            saving_map[saving],
            checking_map[checking],
            credit_amount,
            duration,
            purpose_map[purpose]
        ]).reshape(1, -1)

        prediction = model.predict(data)
        probability = model.predict_proba(data)

        score = round(probability[0][1] * 100, 2)

        if score >= 80:
            risk = "Very Low Risk"
        elif score >= 60:
            risk = "Low Risk"
        elif score >= 40:
            risk = "Moderate Risk"
        else:
            risk = "High Risk"

        r1, r2 = st.columns(2)

        with r1:
            st.metric("Credit Score", f"{score}%")

        with r2:
            st.metric("Risk Category", risk)

        st.subheader("🤖 AI Recommendation")

        if score >= 70:
            st.success("Loan Approval Recommended")
        else:
            st.warning("Manual Verification Required")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Credit Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 40], "color": "red"},
                    {"range": [40, 70], "color": "orange"},
                    {"range": [70, 100], "color": "green"}
                ]
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.container(border=True):

            st.subheader("📋 Assessment Summary")

            st.write(f"**Age:** {age}")
            st.write(f"**Gender:** {sex}")
            st.write(f"**Credit Amount:** ₹{credit_amount:,}")
            st.write(f"**Duration:** {duration} Months")
            st.write(f"**Risk Level:** {risk}")

        report = f"""
Credit Risk Assessment Report

Age: {age}
Gender: {sex}
Credit Amount: {credit_amount}
Duration: {duration} Months

Credit Score: {score}
Risk Category: {risk}
"""

        st.download_button(
            "📄 Download Report",
            report,
            file_name="Credit_Report.txt"
        )

        if prediction[0] == 1:
            st.success("✅ Low Credit Risk — Loan Recommended")
        else:
            st.error("⚠️ High Credit Risk — Further Review Recommended")

# ====================================================
# ABOUT PAGE
# ====================================================

elif page == "About":

    st.title("📖 About CreditSensei AI")

    st.write("""
CreditSensei AI is a Machine Learning based Credit Risk Prediction System.

### Technologies Used
- Python
- Streamlit
- Scikit-Learn
- Plotly
- Random Forest Classifier

### Dataset
German Credit Dataset

### Model Performance
- Accuracy: 78%
- Precision: 80.0%
- Recall: 91.4%
- F1 Score: 85.3%
- ROC-AUC: 77.0%

### Features
- Credit Risk Prediction
- AI Recommendation System
- Credit Score Gauge
- Downloadable Report
- Professional Dashboard
""")
    st.subheader("Project Objective")

st.write("""
The objective of CreditSensei AI is to predict customer
creditworthiness using Machine Learning techniques and
assist financial institutions in loan approval decisions.
""")

st.success("Developed by Sadhana Patil")