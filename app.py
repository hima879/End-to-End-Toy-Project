import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-top: 0;
    }
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4f46e5;
        background: rgba(79, 70, 229, 0.08);
        margin-top: 0.5rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Model ----------
@st.cache_resource
def load_model():
    df = pd.read_csv("placement.csv")
    df = df.iloc[:, 1:]  # drop unnamed index col
    X = df.iloc[:, 0:2]  # cgpa, iq
    y = df.iloc[:, -1]   # placement

    # Wrap in a pipeline so scaling is baked in (better for LogisticRegression)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)
    return pipe, X.columns.tolist(), df


model, feature_names, train_df = load_model()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** Logistic Regression (with StandardScaler)
        **Features:** CGPA, IQ
        **Target:** Placement (0 = Not placed, 1 = Placed)
        """
    )

    with st.expander("📊 Training data summary"):
        st.write(f"**Samples:** {len(train_df)}")
        st.dataframe(train_df.describe().T, use_container_width=True)

    with st.expander("🎯 Model coefficients"):
        clf = model.named_steps["clf"]
        coefs = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": clf.coef_[0],
        })
        st.dataframe(coefs, use_container_width=True, hide_index=True)
        st.caption("Larger magnitude = stronger influence on the prediction.")

    st.markdown("---")
    st.markdown(
        """
        **How to use:**
        1. Enter CGPA and IQ (or drag the sliders).
        2. The prediction updates live.
        3. Check the confidence bars below.
        """
    )


# ---------- Header ----------
st.markdown('<p class="main-title">🎓 Student Placement Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Predict placement likelihood from CGPA and IQ using Logistic Regression.</p>',
    unsafe_allow_html=True,
)
st.divider()


# ---------- Inputs (with sliders + number inputs) ----------
st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.1,
        help="Slide to pick a CGPA between 0 and 10.",
    )

with col2:
    iq = st.slider(
        "IQ",
        min_value=50,
        max_value=200,
        value=120,
        step=1,
        help="Slide to pick an IQ score.",
    )


# ---------- Live prediction ----------
features = pd.DataFrame([[cgpa, iq]], columns=feature_names)
prediction = int(model.predict(features)[0])
proba = model.predict_proba(features)[0]
placed_prob = float(proba[1])
not_placed_prob = float(proba[0])

st.divider()
st.subheader("📊 Prediction")

# Top row: verdict + confidence metric
vcol, mcol = st.columns([2, 1])

with vcol:
    if prediction == 1:
        st.success(f"✅ Likely to be **PLACED** (confidence: {placed_prob:.1%})")
    else:
        st.error(f"❌ Likely to be **NOT PLACED** (confidence: {not_placed_prob:.1%})")

with mcol:
    st.metric("Placement probability", f"{placed_prob:.1%}")

# Confidence bars
st.markdown("#### Confidence breakdown")
st.write(f"**Placed** — {placed_prob:.2%}")
st.progress(placed_prob)

st.write(f"**Not placed** — {not_placed_prob:.2%}")
st.progress(not_placed_prob)

# Decorative confetti only on strong positive predictions
if prediction == 1 and placed_prob >= 0.75:
    st.balloons()
elif prediction == 0 and not_placed_prob >= 0.75:
    st.toast("Low placement probability detected.", icon="⚠️")


# ---------- Input sensitivity (nice interaction) ----------
with st.expander("🔍 How does each feature affect this prediction?"):
    # Simple one-at-a-time perturbation to show local sensitivity
    base_prob = placed_prob

    cgpa_lo = max(0.0, cgpa - 1.0)
    cgpa_hi = min(10.0, cgpa + 1.0)
    iq_lo = max(50, iq - 10)
    iq_hi = min(200, iq + 10)

    cgpa_lo_p = model.predict_proba(pd.DataFrame([[cgpa_lo, iq]], columns=feature_names))[0][1]
    cgpa_hi_p = model.predict_proba(pd.DataFrame([[cgpa_hi, iq]], columns=feature_names))[0][1]
    iq_lo_p = model.predict_proba(pd.DataFrame([[cgpa, iq_lo]], columns=feature_names))[0][1]
    iq_hi_p = model.predict_proba(pd.DataFrame([[cgpa, iq_hi]], columns=feature_names))[0][1]

    sens = pd.DataFrame({
        "Scenario": [
            f"CGPA {cgpa_lo:.1f} (current −1)",
            f"Current (CGPA {cgpa:.1f}, IQ {iq})",
            f"CGPA {cgpa_hi:.1f} (current +1)",
            f"IQ {iq_lo} (current −10)",
            f"IQ {iq_hi} (current +10)",
        ],
        "Placement probability": [
            cgpa_lo_p, base_prob, cgpa_hi_p, iq_lo_p, iq_hi_p,
        ],
    })
    sens["Δ vs current"] = sens["Placement probability"] - base_prob
    st.dataframe(
        sens.style.format({
            "Placement probability": "{:.1%}",
            "Δ vs current": "{:+.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Each row shows what happens if only that feature changes, "
        "keeping everything else the same."
    )


# ---------- Quick presets ----------
st.markdown("#### ⚡ Quick presets")
p1, p2, p3 = st.columns(3)
p1.info("**Strong profile** — CGPA 8.5, IQ 130")
p2.info("**Average profile** — CGPA 6.5, IQ 110")
p3.info("**Weak profile** — CGPA 4.5, IQ 90")
st.caption("Move the sliders above to match these presets to explore outcomes.")
