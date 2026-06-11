import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="NailVision AI",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #1e293b
    );
}

html, body, [class*="css"]{
    color:white;
}

.hero{
    text-align:center;
    padding-top:20px;
    padding-bottom:20px;
}

.hero-title{
    font-size:4.5rem;
    font-weight:800;

    background: linear-gradient(
        90deg,
        #60a5fa,
        #22d3ee,
        #a78bfa
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-sub{
    color:#cbd5e1;
    font-size:1.3rem;
}

.card{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(15px);

    border-radius:20px;
    padding:20px;

    border:1px solid rgba(255,255,255,0.15);

    margin-bottom:20px;
}

.card:hover{
    transform:translateY(-3px);
    transition:.3s;
}

.metric-container{
    text-align:center;
}

section[data-testid="stSidebar"]{
    background:#0f172a;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/nail_model.h5")

model = load_model()

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = [
    "Acral Lentiginous Melanoma",
    "Onychogryphosis",
    "Blue Finger",
    "Clubbing",
    "Healthy",
    "Pitting"
]

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/artificial-intelligence.png",
        width=80
    )

    selected = option_menu(
        menu_title="NailVision AI",
        options=[
            "Home",
            "Live Demo",
            "Results",
            "Skills",
            "About Me"
        ],
        icons=[
            "house",
            "camera",
            "bar-chart",
            "cpu",
            "person-circle"
        ],
        default_index=0
    )

# =====================================================
# HOME
# =====================================================

if selected == "Home":

    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>
            NailVision AI
        </div>

        <div class='hero-sub'>
            Deep Learning Powered Nail Disease Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "89%"
        )

    with col2:
        st.metric(
            "Conditions",
            "6"
        )

    with col3:
        st.metric(
            "Training Images",
            "3744"
        )

    with col4:
        st.metric(
            "F1 Score",
            "0.89"
        )

    st.markdown("---")

    st.markdown("""
    <div class='card'>

    <h2>Project Overview</h2>

    NailVision AI is a deep-learning image classification system
    developed using TensorFlow, OpenCV, and MobileNetV2.

    The model analyzes uploaded nail images and predicts one of
    six nail conditions.

    Features:

    • Deep Learning Classification

    • Computer Vision Analysis

    • Confidence Scoring

    • Explainable AI (Grad-CAM)

    • Interactive Web Application

    </div>
    """, unsafe_allow_html=True)

    if os.path.exists("gradcam.png"):

        st.subheader("Grad-CAM Explainability")

        st.image(
            "gradcam.png",
            caption="Example AI Attention Map",
            use_container_width=True
        )

# =====================================================
# LIVE DEMO
# =====================================================

elif selected == "Live Demo":

    st.title("🔍 Live Nail Analysis")

    uploaded_file = st.file_uploader(
        "Upload a nail image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1,col2 = st.columns([1,1])

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        img = np.array(image)

        img = cv2.resize(
            img,
            (224,224)
        )

        img = img.astype("float32") / 255.0

        img = np.expand_dims(
            img,
            axis=0
        )

        predictions = model.predict(
            img,
            verbose=0
        )[0]

        predicted_idx = np.argmax(predictions)

        predicted_class = CLASS_NAMES[
            predicted_idx
        ]

        confidence = (
            predictions[predicted_idx]
            * 100
        )

        with col2:

            st.success(
                f"Prediction: {predicted_class}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            confidence_df = pd.DataFrame({
                "Condition": CLASS_NAMES,
                "Confidence": predictions * 100
            })

            fig = px.bar(
                confidence_df,
                x="Condition",
                y="Confidence",
                title="Prediction Confidence"
            )

            fig.update_layout(
                template="plotly_dark"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

# =====================================================
# RESULTS
# =====================================================

elif selected == "Results":

    st.title("📊 Model Results")

    col1,col2 = st.columns(2)

    with col1:

        st.markdown("""
        ### Performance Summary

        - Validation Accuracy: 89%
        - Weighted F1 Score: 0.89
        - 3744 Training Images
        - 91 Validation Images
        - Six Disease Categories
        """)

    with col2:

        if os.path.exists(
            "confusion_matrix.png"
        ):

            st.image(
                "confusion_matrix.png",
                caption="Confusion Matrix",
                use_container_width=True
            )

    st.subheader(
        "Classification Report"
    )

    report_df = pd.DataFrame({

        "Condition":[
            "Melanoma",
            "Onychogryphosis",
            "Blue Finger",
            "Clubbing",
            "Healthy",
            "Pitting"
        ],

        "Precision":[
            0.90,
            0.90,
            0.67,
            1.00,
            1.00,
            0.84
        ],

        "Recall":[
            1.00,
            0.75,
            0.89,
            0.62,
            1.00,
            1.00
        ],

        "F1 Score":[
            0.95,
            0.82,
            0.76,
            0.77,
            1.00,
            0.91
        ]
    })

    st.dataframe(
        report_df,
        use_container_width=True
    )

# =====================================================
# SKILLS
# =====================================================

elif selected == "Skills":

    st.title("🛠 Technical Skills")

    st.markdown("""
    <div class='card'>

    <h2>Programming</h2>

    • Python

    • NumPy

    • Pandas

    • Object-Oriented Programming

    </div>

    <div class='card'>

    <h2>Machine Learning</h2>

    • TensorFlow

    • Deep Learning

    • MobileNetV2

    • Transfer Learning

    • Neural Networks

    </div>

    <div class='card'>

    <h2>Computer Vision</h2>

    • OpenCV

    • Image Classification

    • Image Processing

    • Grad-CAM

    </div>

    <div class='card'>

    <h2>Software Development</h2>

    • Streamlit

    • Git

    • GitHub

    • Deployment

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# ABOUT ME
# =====================================================

elif selected == "About Me":

    st.title("👨‍💻 About Me")

    st.markdown("""
    <div class='card'>

    <h2>Student Developer</h2>

    I am a high school student interested in:

    • Artificial Intelligence

    • Machine Learning

    • Data Science

    • Computer Vision

    • Electrical Engineering

    This project was created to explore how
    deep learning can identify visible nail
    abnormalities from images.

    </div>
    """, unsafe_allow_html=True)

    st.subheader(
        "Development Journey"
    )

    st.markdown("""
    ### Dataset Collection
    Organized and cleaned 3744 training images.

    ### Model Training
    Built a MobileNetV2-based classifier using transfer learning.

    ### Evaluation
    Generated confusion matrices and classification reports.

    ### Explainability
    Implemented Grad-CAM visualizations.

    ### Deployment
    Developed and deployed an interactive Streamlit web application.
    """)

    st.subheader(
        "Future Improvements"
    )

    st.markdown("""
    - Larger dataset

    - More nail conditions

    - Mobile application

    - TensorFlow Lite deployment

    - Raspberry Pi integration

    - Real-time camera detection
    """)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "NailVision AI • TensorFlow • OpenCV • Streamlit • MobileNetV2"
)