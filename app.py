import os
import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="NailVision AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CONSTANTS
# =====================================================
MODEL_PATH = "models/nail_model.h5"
SAMPLE_DIR = "sample_nails"

CLASS_NAMES = [
    "Acral Lentiginous Melanoma",
    "Onychogryphosis",
    "Blue Finger",
    "Clubbing",
    "Healthy",
    "Pitting"
]

REPORT_DF = pd.DataFrame({
    "Condition": ["Melanoma", "Onychogryphosis", "Blue Finger", "Clubbing", "Healthy", "Pitting"],
    "Precision": [0.90, 0.90, 0.67, 1.00, 1.00, 0.84],
    "Recall": [1.00, 0.75, 0.89, 0.62, 1.00, 1.00],
    "F1 Score": [0.95, 0.82, 0.76, 0.77, 1.00, 0.91],
    "Support": [18, 12, 9, 16, 20, 16]
})

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #070B16;
    --card: rgba(15, 23, 42, 0.84);
    --card-2: rgba(30, 41, 59, 0.72);
    --border: rgba(148, 163, 184, 0.22);
    --text: #F8FAFC;
    --muted: #CBD5E1;
    --muted-2: #94A3B8;
    --cyan: #38BDF8;
    --blue: #3B82F6;
    --purple: #8B5CF6;
    --pink: #EC4899;
    --green: #22C55E;
    --amber: #F59E0B;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(56,189,248,0.18), transparent 25%),
        radial-gradient(circle at 85% 12%, rgba(139,92,246,0.18), transparent 25%),
        radial-gradient(circle at 50% 90%, rgba(236,72,153,0.10), transparent 30%),
        linear-gradient(135deg, #070B16 0%, #0B1120 45%, #111827 100%);
    background-attachment: fixed;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.34;
    background-image:
      radial-gradient(circle, rgba(255,255,255,.18) 1px, transparent 1px),
      radial-gradient(circle, rgba(56,189,248,.16) 1px, transparent 1px);
    background-size: 42px 42px, 77px 77px;
    animation: drift 24s linear infinite;
    z-index: 0;
}

@keyframes drift {
    0% { background-position: 0 0, 0 0; }
    100% { background-position: 420px 260px, -390px 310px; }
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 1.25rem; max-width: 1240px; position: relative; z-index: 2; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: rgba(2, 6, 23, 0.62);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 999px;
    padding: 8px;
    backdrop-filter: blur(18px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    white-space: pre-wrap;
    border-radius: 999px;
    color: #CBD5E1;
    padding: 0 18px;
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(56,189,248,.25), rgba(139,92,246,.28));
    color: #FFFFFF !important;
    border: 1px solid rgba(56,189,248,.35);
}

/* Text readability */
p, li, span, div { color: inherit; }
.readable, .readable p, .readable li {
    color: #E2E8F0 !important;
    line-height: 1.75;
    font-size: 1.02rem;
}
.muted { color: #CBD5E1 !important; }
.small-muted { color: #94A3B8 !important; font-size: 0.94rem; }

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 76px 32px 52px 32px;
    border-radius: 34px;
    border: 1px solid rgba(148, 163, 184, 0.22);
    background:
        linear-gradient(135deg, rgba(15,23,42,.92), rgba(30,41,59,.55)),
        radial-gradient(circle at 20% 20%, rgba(56,189,248,.18), transparent 28%),
        radial-gradient(circle at 80% 20%, rgba(139,92,246,.20), transparent 28%);
    box-shadow: 0 25px 90px rgba(0,0,0,.38);
    backdrop-filter: blur(22px);
    margin-bottom: 28px;
}
.hero::before {
    content: "";
    position: absolute;
    width: 420px;
    height: 420px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,.25), transparent 65%);
    top: -180px;
    left: -120px;
    animation: floatOrb 8s ease-in-out infinite;
}
.hero::after {
    content: "";
    position: absolute;
    width: 480px;
    height: 480px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139,92,246,.22), transparent 65%);
    right: -160px;
    bottom: -220px;
    animation: floatOrb 10s ease-in-out infinite reverse;
}
@keyframes floatOrb {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(24px, -18px) scale(1.08); }
}
.hero-content { position: relative; z-index: 2; text-align: center; }
.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border: 1px solid rgba(56,189,248,.32);
    border-radius: 999px;
    background: rgba(56,189,248,.10);
    color: #BAE6FD !important;
    font-weight: 800;
    margin-bottom: 18px;
}
.hero-title {
    font-size: clamp(3.3rem, 8vw, 7.2rem);
    font-weight: 950;
    line-height: .95;
    letter-spacing: -0.07em;
    margin: 0;
    background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 35%, #A78BFA 70%, #F0ABFC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 28px rgba(56,189,248,.28));
}
.hero-subtitle {
    max-width: 850px;
    margin: 22px auto 0 auto;
    color: #DCEBFF !important;
    font-size: 1.18rem;
    line-height: 1.75;
}
.hero-actions { margin-top: 28px; display: flex; justify-content: center; gap: 14px; flex-wrap: wrap; }
.fake-button {
    display: inline-block;
    padding: 14px 20px;
    border-radius: 999px;
    font-weight: 900;
    border: 1px solid rgba(255,255,255,.14);
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: #FFFFFF !important;
    box-shadow: 0 12px 28px rgba(37,99,235,.25);
}
.fake-button.secondary {
    background: rgba(15,23,42,.68);
    color: #E0F2FE !important;
    border: 1px solid rgba(56,189,248,.28);
}

/* Cards */
.glass-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 26px;
    box-shadow: 0 18px 55px rgba(0,0,0,0.32);
    backdrop-filter: blur(20px);
    transition: transform .25s ease, box-shadow .25s ease, border .25s ease;
    margin-bottom: 22px;
}
.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56,189,248,.36);
    box-shadow: 0 0 40px rgba(56,189,248,.16), 0 18px 55px rgba(0,0,0,0.38);
}
.card-title {
    font-size: 1.35rem;
    font-weight: 900;
    margin-bottom: 10px;
    color: #F8FAFC !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 22px;
    padding: 18px 18px;
    box-shadow: 0 12px 35px rgba(0,0,0,.28);
}
[data-testid="stMetricLabel"] p { color: #CBD5E1 !important; font-weight: 800; }
[data-testid="stMetricValue"] { color: #F8FAFC !important; font-weight: 950; }
[data-testid="stMetricDelta"] { color: #67E8F9 !important; }

/* Badges */
.badge {
    display: inline-block;
    padding: 9px 14px;
    border-radius: 999px;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.34);
    margin: 5px;
    color: #E0F2FE !important;
    font-weight: 800;
}
.badge.purple { background: rgba(139,92,246,0.13); border-color: rgba(139,92,246,.38); color: #EDE9FE !important; }
.badge.green { background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,.35); color: #DCFCE7 !important; }

/* Upload */
.upload-panel {
    background: linear-gradient(135deg, rgba(15,23,42,.94), rgba(30,41,59,.66));
    border: 1px solid rgba(56,189,248,.22);
    border-radius: 30px;
    padding: 30px;
    box-shadow: 0 18px 55px rgba(0,0,0,.32);
    margin-bottom: 18px;
}
.result-card {
    background: linear-gradient(135deg, rgba(14,165,233,.16), rgba(124,58,237,.17));
    border: 1px solid rgba(56,189,248,.32);
    border-radius: 28px;
    padding: 28px;
    box-shadow: 0 0 44px rgba(56,189,248,.14);
}
.big-result {
    font-size: clamp(2rem, 4vw, 3rem);
    line-height: 1.1;
    font-weight: 950;
    color: #67E8F9 !important;
    text-shadow: 0 0 20px rgba(56,189,248,.42);
    margin: 8px 0;
}
.warning-note {
    padding: 16px 18px;
    border-radius: 18px;
    background: rgba(245,158,11,.12);
    border: 1px solid rgba(245,158,11,.34);
    color: #FEF3C7 !important;
    font-weight: 700;
}

/* Gallery */
.gallery-img {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,.22);
    box-shadow: 0 12px 30px rgba(0,0,0,.28);
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(56,189,248,0.42);
    background: linear-gradient(90deg, #2563EB, #7C3AED);
    color: #FFFFFF;
    font-weight: 900;
    padding: .65rem 1.1rem;
    box-shadow: 0 12px 26px rgba(37,99,235,.22);
    transition: transform .2s ease, box-shadow .2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(56,189,248,.32);
    color: white;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(148,163,184,.22);
}

.footer {
    text-align: center;
    color: #94A3B8 !important;
    padding: 28px 0 14px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL + HELPERS
# =====================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


def predict_pil(image: Image.Image):
    if model is None:
        return None, None
    img = image.convert("RGB")
    arr = np.array(img)
    arr = cv2.resize(arr, (224, 224))
    arr = arr.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    predictions = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(predictions))
    return idx, predictions


def confidence_chart(predictions, title="Prediction Confidence"):
    df = pd.DataFrame({"Condition": CLASS_NAMES, "Confidence": predictions * 100}).sort_values("Confidence", ascending=True)
    fig = px.bar(
        df,
        x="Confidence",
        y="Condition",
        orientation="h",
        color="Confidence",
        text="Confidence",
        color_continuous_scale="Turbo",
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        template="plotly_dark",
        height=440,
        margin=dict(l=10, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        coloraxis_showscale=False,
        xaxis=dict(range=[0, max(100, float(df["Confidence"].max()) + 8)]),
    )
    return fig


def get_sample_images(limit=24):
    if not os.path.exists(SAMPLE_DIR):
        return []
    files = []
    for f in os.listdir(SAMPLE_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(os.path.join(SAMPLE_DIR, f))
    return sorted(files)[:limit]

# =====================================================
# TOP NAVIGATION
# =====================================================
st.markdown("""
<div class="hero">
  <div class="hero-content">
    <div class="hero-kicker">⚡ Portfolio AI Project</div>
    <h1 class="hero-title">NailVision AI</h1>
    <p class="hero-subtitle">
      A polished computer vision web app that classifies nail images using TensorFlow, OpenCV, MobileNetV2, and an interactive Streamlit dashboard.
    </p>
    <div class="hero-actions">
      <span class="fake-button">Upload & Analyze</span>
      <span class="fake-button secondary">View Model Results</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

home_tab, demo_tab, results_tab, gallery_tab, tech_tab, about_tab = st.tabs([
    "🏠 Home",
    "🔍 Live Demo",
    "📊 Results",
    "🖼️ Sample Gallery",
    "🛠 Technology",
    "👨‍💻 About Me",
])

# =====================================================
# HOME
# =====================================================
with home_tab:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Validation Accuracy", "89%")
    col2.metric("Classes", "6")
    col3.metric("Training Images", "3,744")
    col4.metric("Weighted F1", "0.89")

    c1, c2 = st.columns([1.15, .85])
    with c1:
        st.markdown("""
        <div class="glass-card readable">
          <div class="card-title">🚀 Project Overview</div>
          <p>
          NailVision AI is an end-to-end machine learning project that uses computer vision to classify six nail condition categories from images.
          It includes a trained TensorFlow model, image preprocessing, evaluation metrics, interactive visualizations, and a deployed web interface.
          </p>
          <span class="badge">TensorFlow</span>
          <span class="badge purple">OpenCV</span>
          <span class="badge">MobileNetV2</span>
          <span class="badge green">Streamlit</span>
          <span class="badge purple">Plotly</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card readable">
          <div class="card-title">⚠️ Medical Disclaimer</div>
          <p>
          This app is an educational AI project. It does not provide medical advice, diagnosis, or treatment.
          Any real health concern should be evaluated by a qualified medical professional.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card readable">
      <div class="card-title">What Makes This Project Strong?</div>
      <ul>
        <li>Uses transfer learning instead of a basic from-scratch model.</li>
        <li>Evaluates performance with accuracy, precision, recall, F1-score, and confusion matrix support.</li>
        <li>Includes a real web app users can interact with.</li>
        <li>Shows both engineering and data science skills.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# LIVE DEMO
# =====================================================
with demo_tab:
    st.markdown("""
    <div class="upload-panel readable">
      <div class="card-title">📤 Upload a Nail Image</div>
      <p>Upload a JPG or PNG image. The model will return the predicted nail condition and confidence scores.</p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error(f"Model not found at {MODEL_PATH}. Make sure your file is in models/nail_model.h5")
    else:
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            with st.spinner("Analyzing image with NailVision AI..."):
                idx, predictions = predict_pil(image)
            confidence = predictions[idx] * 100

            left, right = st.columns([.92, 1.08])
            with left:
                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.markdown("<div class='warning-note'>Educational output only — not medical diagnosis.</div>", unsafe_allow_html=True)
            with right:
                st.markdown(f"""
                <div class="result-card">
                  <div class="card-title">Prediction Result</div>
                  <div class="big-result">{CLASS_NAMES[idx]}</div>
                  <h3 style="color:#F8FAFC;">Confidence: {confidence:.2f}%</h3>
                  <p class="small-muted">The chart below shows the model's confidence across all classes.</p>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(confidence_chart(predictions), use_column_width=True)
        else:
            st.info("Upload an image to run a prediction.")

# =====================================================
# RESULTS
# =====================================================
with results_tab:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "89%")
    col2.metric("Macro Avg F1", "0.87")
    col3.metric("Weighted F1", "0.89")
    col4.metric("Validation Images", "91")

    st.markdown("<div class='glass-card readable'><div class='card-title'>📈 Interactive Performance Dashboard</div><p>Select a metric to compare model performance by class.</p></div>", unsafe_allow_html=True)

    metric_choice = st.selectbox("Metric", ["Precision", "Recall", "F1 Score"], index=2)
    fig_metric = px.bar(
        REPORT_DF,
        x="Condition",
        y=metric_choice,
        color=metric_choice,
        text=metric_choice,
        color_continuous_scale="Viridis",
        title=f"{metric_choice} by Class",
    )
    fig_metric.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_metric.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_metric, use_column_width=True)

    left, right = st.columns([1, 1])
    with left:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=REPORT_DF["F1 Score"],
            theta=REPORT_DF["Condition"],
            fill="toself",
            name="F1 Score",
            line=dict(color="#38BDF8", width=3),
            fillcolor="rgba(56,189,248,.22)",
        ))
        fig_radar.update_layout(
            title="F1 Score Radar",
            template="plotly_dark",
            height=500,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
        )
        st.plotly_chart(fig_radar, use_column_width=True)
    with right:
        st.markdown("<div class='glass-card readable'><div class='card-title'>📋 Classification Report</div><p>Precision shows how reliable a prediction is. Recall shows how many true examples were found. F1 balances both.</p></div>", unsafe_allow_html=True)
        st.dataframe(REPORT_DF, use_container_width=True, hide_index=True)

    if os.path.exists("confusion_matrix.png"):
        st.subheader("Confusion Matrix")
        st.image("confusion_matrix.png", use_column_width=True)
    else:
        st.info("Optional: add confusion_matrix.png to the project folder to display it here.")

# =====================================================
# SAMPLE GALLERY
# =====================================================
with gallery_tab:
    st.markdown("<div class='glass-card readable'><div class='card-title'>🖼️ Sample Nail Gallery</div><p>Add sample images to a folder named <b>sample_nails</b>. Click Analyze under any image to run the model.</p></div>", unsafe_allow_html=True)
    samples = get_sample_images(limit=24)
    if not samples:
        st.info("No sample images found. Create a folder named sample_nails and add JPG/PNG nail images.")
    else:
        cols = st.columns(4)
        for i, path in enumerate(samples):
            with cols[i % 4]:
                try:
                    img = Image.open(path).convert("RGB")
                    st.image(img, use_column_width=True)
                    st.caption(Path(path).name)
                    if st.button("Analyze", key=f"analyze_{i}"):
                        if model is None:
                            st.error("Model not loaded.")
                        else:
                            idx, predictions = predict_pil(img)
                            st.session_state["gallery_prediction"] = (path, idx, predictions)
                except Exception as e:
                    st.warning(f"Could not load {path}: {e}")

        if "gallery_prediction" in st.session_state:
            path, idx, predictions = st.session_state["gallery_prediction"]
            st.markdown("---")
            st.subheader("Selected Sample Prediction")
            c1, c2 = st.columns([.9, 1.1])
            with c1:
                st.image(Image.open(path).convert("RGB"), caption=Path(path).name, use_column_width=True)
            with c2:
                confidence = predictions[idx] * 100
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-title">Prediction</div>
                    <div class="big-result">{CLASS_NAMES[idx]}</div>
                    <h3 style="color:#F8FAFC;">Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(confidence_chart(predictions, "Sample Confidence Scores"), use_column_width=True)

# =====================================================
# TECHNOLOGY
# =====================================================
with tech_tab:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card readable">
          <div class="card-title">🐍 Programming</div>
          <p>Python, NumPy, Pandas, project organization, model integration.</p>
        </div>
        <div class="glass-card readable">
          <div class="card-title">🧠 Machine Learning</div>
          <p>TensorFlow, Keras, transfer learning, MobileNetV2, softmax classification.</p>
        </div>
        <div class="glass-card readable">
          <div class="card-title">📷 Computer Vision</div>
          <p>OpenCV, image resizing, normalization, image preprocessing, visual classification.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card readable">
          <div class="card-title">📊 Data Science</div>
          <p>Accuracy, precision, recall, F1-score, confusion matrix, validation analysis.</p>
        </div>
        <div class="glass-card readable">
          <div class="card-title">🌐 Web Development</div>
          <p>Streamlit UI, Plotly dashboards, deployment-ready app structure.</p>
        </div>
        <div class="glass-card readable">
          <div class="card-title">🚀 Deployment</div>
          <p>requirements.txt, runtime configuration, Streamlit Cloud deployment workflow.</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# ABOUT
# =====================================================
with about_tab:
    st.markdown("""
    <div class="glass-card readable">
      <div class="card-title">👨‍💻 About Me</div>
      <p>
      I am a high school student interested in artificial intelligence, data science, computer vision, and electrical engineering.
      This project helped me learn how to build a full AI workflow from dataset preparation to model training, evaluation, and web deployment.
      </p>
    </div>
    """, unsafe_allow_html=True)

    timeline = pd.DataFrame({
        "Stage": ["Dataset", "Model", "Evaluation", "Website", "Deployment"],
        "What I Did": [
            "Organized nail images into six classes.",
            "Trained a MobileNetV2 transfer-learning classifier.",
            "Generated accuracy, F1-score, and class-level metrics.",
            "Built an interactive Streamlit web app.",
            "Prepared the app for public hosting."
        ]
    })
    st.dataframe(timeline, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="glass-card readable">
      <div class="card-title">Future Improvements</div>
      <ul>
        <li>Increase validation data size for more reliable results.</li>
        <li>Add more nail condition categories.</li>
        <li>Add Grad-CAM directly inside the app.</li>
        <li>Create a mobile-friendly version.</li>
        <li>Explore TensorFlow Lite or Raspberry Pi deployment.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  NailVision AI • TensorFlow • OpenCV • Streamlit • MobileNetV2
</div>
""", unsafe_allow_html=True)
