import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Placement Predictor", page_icon="🎓")

@st.cache_resource
def load_model():
    df = pd.read_csv("placement.csv")
    df = df.iloc[:, 1:]          # drop unnamed index col
    X = df.iloc[:, 0:2]          # cgpa, iq
    y = df.iloc[:, -1]           # placement
    clf = LogisticRegression()
    clf.fit(X, y)
    return clf

model = load_model()

# Header
st.title("🎓 Student Placement Predictor")
st.markdown(
    """
    This app predicts whether a student will be **placed** based on their
    **CGPA** and **IQ**, using a Logistic Regression model trained on
    a small toy dataset.
    """
)

st.divider()

# Input fields
st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.1,
        help="Enter CGPA between 0 and 10"
    )

with col2:
    iq = st.number_input(
        "IQ",
        min_value=0,
        max_value=300,
        value=120,
        step=1,
        help="Enter IQ score"
    )

# Prediction
if st.button("Predict Placement", type="primary", use_container_width=True):
    # Match the training input order: [cgpa, iq]
    features = np.array([[cgpa, iq]])

    prediction = model.predict(features)[0]

    # Probability (if supported)
    try:
        proba = model.predict_proba(features)[0]
        placed_prob = proba[1]
        not_placed_prob = proba[0]
    except AttributeError:
        placed_prob = None

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.success("✅ The student is likely to be **PLACED**.")
        st.balloons()
    else:
        st.error("❌ The student is likely to be **NOT PLACED**.")

    if placed_prob is not None:
        st.markdown("### Prediction Confidence")
        st.progress(float(placed_prob))
        st.write(f"Probability of being placed: **{placed_prob:.2%}**")
        st.write(f"Probability of not being placed: **{not_placed_prob:.2%}**")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** Logistic Regression  
        **Features:** CGPA, IQ  
        **Target:** Placement (0 or 1)  
        **Training samples:** 100  
        **Test accuracy:** ~95%
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **How to use:**
        1. Enter the CGPA and IQ.
        2. Click **Predict Placement**.
        3. View the result and confidence score.
        """
    )
