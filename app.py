import streamlit as st
import cv2
import pandas as pd
from PIL import Image
import numpy as np

from src.pipeline import WNIRPipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="WNIR | Railway Recognition",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f4f6f9;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #142d4e;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #142d4e;
        font-size: 21px;
        font-weight: 700;
        margin-top: 15px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🚆 Wagon Number and Container ID Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered detection and text recognition system for freight trains'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_pipeline():
    return WNIRPipeline()


pipeline = load_pipeline()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    st.divider()

    st.markdown("### About WNIR")

    st.write(
        "WNIR detects wagon number and container code "
        "regions and extracts alphanumeric text from them."
    )

    st.markdown("**Technologies**")
    st.write("- YOLOv8")
    st.write("- EasyOCR")
    st.write("- OpenCV")
    st.write("- Streamlit")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📤 Upload Freight Train Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a freight train image for wagon and container recognition."
)


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.success("Image uploaded successfully!")

    if st.button("🔍 Detect and Recognize", type="primary", use_container_width=True):

        with st.spinner("Detecting regions and recognizing text..."):

            # Convert RGB image to OpenCV BGR format
            opencv_image = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

            # Run WNIR pipeline
            annotated_image, results_df = pipeline.process(
                opencv_image,
                confidence=confidence
            )

            # Convert annotated image back to RGB
            annotated_rgb = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )

        st.success("Recognition completed!")

        # ------------------------------------------
        # IMAGE COMPARISON
        # ------------------------------------------

        st.markdown(
            '<div class="section-title">🖼️ Detection Visualization</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(
                image_array,
                use_container_width=True
            )

        with col2:
            st.subheader("Detected Image")
            st.image(
                annotated_rgb,
                use_container_width=True
            )

        # ------------------------------------------
        # RESULTS SUMMARY
        # ------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Recognition Summary</div>',
            unsafe_allow_html=True
        )

        if results_df is None:
            results_df = pd.DataFrame()

        total = len(results_df)

        if total > 0:
            wagon_count = (
                results_df["category"] == "Wagon Number"
            ).sum()

            container_count = (
                results_df["category"] == "Container Code"
            ).sum()
        else:
            wagon_count = 0
            container_count = 0

        m1, m2, m3 = st.columns(3)

        m1.metric("Total Detections", total)
        m2.metric("Wagon Numbers", int(wagon_count))
        m3.metric("Container IDs", int(container_count))

        # ------------------------------------------
        # RESULTS TABLE
        # ------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">📋 Extracted Information</div>',
            unsafe_allow_html=True
        )

        if total > 0:

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            # CSV DOWNLOAD
            csv_data = results_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Recognition Results",
                data=csv_data,
                file_name="wnir_recognition_results.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            st.warning(
                "No wagon numbers or container codes were detected. "
                "Try another image or lower the confidence threshold."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="footer">'
    'WNIR | Automatic Wagon Number and Container ID Recognition System'
    '<br>Powered by YOLOv8, EasyOCR and OpenCV'
    '</div>',
    unsafe_allow_html=True
)
