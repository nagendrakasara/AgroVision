
from pathlib import Path
from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO


# =========================================================
# 1. CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AgroVision | Smart Farming",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = Path(__file__).resolve().parent.parent

MOBILENET_PATH = BASE / "models" / "plant_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = BASE / "models" / "plant_disease_class_names.json"
VALIDATOR_PATH = BASE / "models" / "plant_leaf_validator.keras"
VALIDATOR_CONFIG_PATH = BASE / "models" / "plant_leaf_validator_config.json"

YOLO_PATH = (
    BASE / "outputs" / "yolo_training" /
    "disease_localization_v1" / "weights" / "best.pt"
)

LOCATIONS = {
    # Telangana
    "Bhongir, Telangana": (17.51, 78.89),
    "Hyderabad, Telangana": (17.3850, 78.4867),
    "Warangal, Telangana": (17.9689, 79.5941),
    "Karimnagar, Telangana": (18.4386, 79.1288),
    "Nalgonda, Telangana": (17.0575, 79.2684),
    "Khammam, Telangana": (17.2473, 80.1514),
    "Nizamabad, Telangana": (18.6725, 78.0941),
    "Mahbubnagar, Telangana": (16.7370, 78.0081),

    # Andhra Pradesh
    "Vijayawada, Andhra Pradesh": (16.5062, 80.6480),
    "Visakhapatnam, Andhra Pradesh": (17.6868, 83.2185),
    "Guntur, Andhra Pradesh": (16.3067, 80.4365),
    "Tirupati, Andhra Pradesh": (13.6288, 79.4192),
    "Kurnool, Andhra Pradesh": (15.8281, 78.0373),
    "Rajahmundry, Andhra Pradesh": (17.0005, 81.8040),
    "Nellore, Andhra Pradesh": (14.4426, 79.9865),
    "Anantapur, Andhra Pradesh": (14.6819, 77.6006),
}


# =========================================================

# =========================================================
# 2. AGRICULTURAL THEME

# AGROVISION_VISIBILITY_FIX
st.markdown("""
<style>
/* File uploader */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploaderFileName"] {
    color: #173b2a !important;
    opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #f4f8f2 !important;
    border: 1px solid #6b967a !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background-color: #17643d !important;
    color: #ffffff !important;
}

/* Clear status messages */
[data-testid="stAlert"] {
    border-radius: 9px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] {
    color: #172d22 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Date input and selected date */
[data-testid="stDateInput"] label,
[data-testid="stDateInput"] label p {
    color: #173b2a !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stDateInput"] input {
    background-color: #ffffff !important;
    color: #173b2a !important;
    -webkit-text-fill-color: #173b2a !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

[data-testid="stDateInput"] [data-baseweb="input"] {
    background-color: #ffffff !important;
}

[data-testid="stDateInput"] svg {
    color: #17643d !important;
    fill: #17643d !important;
}

/* Calendar popup */
[data-baseweb="calendar"],
[data-baseweb="calendar"] button {
    color: #173b2a !important;
}

[data-baseweb="calendar"] {
    background-color: #ffffff !important;
}

/* Image captions and supporting text */
[data-testid="stImageCaption"],
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: #284b38 !important;
    opacity: 1 !important;
}

/* Standard page text */
[data-testid="stAppViewContainer"] .stMarkdown p {
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
st.markdown("""
<style>
/* Apply the background to Streamlit's actual content layers. */
.stApp {
    background:
        linear-gradient(
            115deg,
            rgba(239, 247, 231, 0.52),
            rgba(250, 250, 240, 0.48),
            rgba(223, 239, 218, 0.50)
        ),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=2000&q=85")
        center / cover fixed;
    color: #173b2b;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: rgba(245, 249, 239, 0.78);
}
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.brand {
    background: linear-gradient(110deg, #103d29, #287b42);
    padding: 28px 32px;
    border-radius: 20px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 7px 24px rgba(20, 65, 42, 0.22);
    border: 1px solid rgba(255,255,255,.35);
}
.brand h1 {
    color: white !important;
    margin: 0;
    font-size: 2.5rem;
}
.brand p {
    color: #e2f3d5 !important;
    margin: 5px 0 0 0;
}
.section-title {
    color: #174b32;
    font-size: 1.65rem;
    font-weight: 750;
    margin: 8px 0;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 247, 0.94);
    border-radius: 16px;
    border: 1px solid rgba(41, 99, 54, .15);
}
div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #176334, #2d8449);
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 650;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #104b29;
    color: white;
    border: 1px solid #8fc69b;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 248, 0.93);
    border-radius: 10px 10px 0 0;
    padding: 12px 18px;
    font-weight: 650;
}
.stTabs [aria-selected="true"] {
    color: #176334;
}
div[data-testid="stMetric"] {
    background: rgba(240, 247, 236, 0.96);
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #d7e8d0;
}
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 248, 0.92);
    padding: 14px;
    border-radius: 14px;
}
.footer {
    text-align: center;
    color: #315c3e;
    padding: 18px 0 5px;
    font-size: 0.9rem;
}

/* Improve contrast of crop and rainfall metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 247, 0.94) !important;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricDelta"] {
    color: #174b32 !important;
    opacity: 1 !important;
}


/* ===== AgroVision: readable dark text ===== */

/* Main text and headings */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    color: #172b20 !important;
}

.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
label {
    color: #172b20 !important;
    font-size: 1.05rem !important;
}

h1, h2, h3, h4,
.section-title {
    color: #123d28 !important;
    font-weight: 800 !important;
}

h1 { font-size: 2.35rem !important; }
h2 { font-size: 1.85rem !important; }
h3 { font-size: 1.5rem !important; }

/* Crop name and all metric values: dark, bold, larger */
[data-testid="stMetric"],
[data-testid="stMetric"] * {
    color: #163d29 !important;
    opacity: 1 !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #244b34 !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #103d29 !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
}

/* Status messages and progress details */
[data-testid="stAlert"] * {
    font-size: 1rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #173b2b !important;
    font-size: 1.05rem !important;
    font-weight: 750 !important;
}
.stTabs [aria-selected="true"] {
    color: #104b29 !important;
}

/* Input labels and values */
[data-testid="stSelectbox"] *,
[data-testid="stDateInput"] *,
[data-testid="stFileUploader"] *,
[data-testid="stNumberInput"] * {
    color: #172b20;
}

/* Footer */
.footer {
    color: #173b2b !important;
    font-size: 1rem !important;
    font-weight: 650 !important;
}

/* Keep the branded header white for contrast on its dark green background */
.brand, .brand h1, .brand p {
    color: #ffffff !important;
}


/* ===== AgroVision: visible dropdown controls ===== */

/* Closed dropdown field */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #398052 !important;
    color: #173b2b !important;
}

/* Selected dropdown text */
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #173b2b !important;
    -webkit-text-fill-color: #173b2b !important;
}

/* Dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] [role="listbox"] {
    background-color: #ffffff !important;
}

/* Individual dropdown options */
div[data-baseweb="popover"] [role="option"] {
    background-color: #ffffff !important;
    color: #173b2b !important;
}

/* Hover and selected options */
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background-color: #e5f2e7 !important;
    color: #174b32 !important;
}

/* Dropdown arrow */
[data-testid="stSelectbox"] svg {
    fill: #174b32 !important;
    color: #174b32 !important;
}


/* ===== FINAL DROPDOWN VISIBILITY FIX ===== */

/* Dropdown closed control */
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1.5px solid #398052 !important;
    color: #173b2b !important;
}

/* Selected state, district and mandal text */
div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stSelectbox"] [data-baseweb="select"] input {
    color: #173b2b !important;
    -webkit-text-fill-color: #173b2b !important;
    opacity: 1 !important;
}

/* Dropdown arrow */
div[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #174b32 !important;
    color: #174b32 !important;
}

/* Open dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"],
ul[role="listbox"] {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

/* Dropdown options */
div[data-baseweb="popover"] [role="option"],
li[role="option"] {
    background: #ffffff !important;
    color: #173b2b !important;
    -webkit-text-fill-color: #173b2b !important;
}

/* Hover and selected option */
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"] {
    background: #e3f2e5 !important;
    color: #14532d !important;
    -webkit-text-fill-color: #14532d !important;
}


/* AgroVision final selectbox contrast */
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stSelectbox"] [role="combobox"] {
    background-color: #ffffff !important;
    color: #173b2b !important;
    border-color: #398052 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] div,
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #173b2b !important;
    -webkit-text-fill-color: #173b2b !important;
    opacity: 1 !important;
}

div[data-baseweb="popover"] [role="option"] {
    background-color: #ffffff !important;
    color: #173b2b !important;
    -webkit-text-fill-color: #173b2b !important;
}

div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background-color: #e3f2e5 !important;
    color: #14532d !important;
}


/* AGROVISION FONT AND DATE VISIBILITY IMPROVEMENTS */

.stApp p,
.stApp label,
.stApp li,
.stApp [data-testid="stCaptionContainer"] {
    font-size: 17px !important;
    color: #173b2a !important;
}

.stApp h1 { font-size: 36px !important; }
.stApp h2 { font-size: 29px !important; }
.stApp h3 { font-size: 23px !important; }

.stButton button {
    font-size: 17px !important;
    font-weight: 700 !important;
    min-height: 46px !important;
}

[data-testid="stWidgetLabel"] p {
    font-size: 17px !important;
    color: #123d29 !important;
    font-weight: 700 !important;
}

[data-testid="stDateInput"] input {
    background: #ffffff !important;
    color: #123d29 !important;
    -webkit-text-fill-color: #123d29 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border: 2px solid #34764d !important;
    min-height: 48px !important;
}

[data-testid="stDateInput"] button {
    background: #ffffff !important;
    color: #17643d !important;
}

[data-testid="stDateInput"] svg {
    color: #17643d !important;
    fill: #17643d !important;
}

[data-baseweb="calendar"] {
    background: #ffffff !important;
    color: #173b2a !important;
    font-size: 16px !important;
}

[data-baseweb="calendar"] button {
    color: #173b2a !important;
    font-size: 16px !important;
}

[data-baseweb="calendar"] button[aria-selected="true"] {
    background: #17643d !important;
    color: #ffffff !important;
}

[data-baseweb="select"] > div {
    font-size: 17px !important;
    color: #173b2a !important;
}

[data-testid="stDataFrame"] {
    font-size: 16px !important;
}


/* AGROVISION FINAL READABILITY UPGRADE */

/* Make the crop field visible earlier in the background image */
.stApp {
    background-position: center 72% !important;
    background-size: cover !important;
    background-attachment: fixed !important;
}

/* General website text */
.stApp,
.stApp p,
.stApp li,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stCaptionContainer"] {
    font-size: 19px !important;
    line-height: 1.55 !important;
    color: #153b29 !important;
}

/* Main title */
.brand h1 {
    font-size: 38px !important;
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* Header subtitle and descriptions */
.brand p {
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Section headings */
.stApp h1 {
    font-size: 38px !important;
}

.stApp h2 {
    font-size: 30px !important;
    color: #123d29 !important;
}

.stApp h3 {
    font-size: 26px !important;
    color: #123d29 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 17px !important;
    font-weight: 700 !important;
    color: #123d29 !important;
}

/* Buttons: white text on green */
.stButton button,
.stDownloadButton button {
    background-color: #176b3d !important;
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    min-height: 50px !important;
    border-radius: 9px !important;
}

.stButton button p,
.stDownloadButton button p {
    color: #ffffff !important;
    font-size: 18px !important;
}

/* Upload area */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] small {
    color: #153b29 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #f7fbf5 !important;
}

/* Dropdowns */
[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"],
[data-baseweb="select"] div {
    font-size: 17px !important;
}

/* Date picker: correct the actual input and its text */
[data-testid="stDateInput"] input,
[data-baseweb="datepicker"] input,
[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #123d29 !important;
    -webkit-text-fill-color: #123d29 !important;
    font-size: 19px !important;
    font-weight: 800 !important;
    min-height: 48px !important;
    border: 2px solid #247346 !important;
    opacity: 1 !important;
}

[data-testid="stDateInput"] button,
[data-testid="stDateInput"] svg {
    color: #176b3d !important;
    fill: #176b3d !important;
}

/* Calendar popup */
[data-baseweb="calendar"] {
    background: #ffffff !important;
    color: #123d29 !important;
    font-size: 17px !important;
}

[data-baseweb="calendar"] button {
    color: #123d29 !important;
    font-size: 17px !important;
}

[data-baseweb="calendar"] button[aria-selected="true"] {
    background: #176b3d !important;
    color: #ffffff !important;
}

/* Status messages */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    font-size: 17px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stAlert"][kind="error"] p {
    color: #8b1e1e !important;
}

[data-testid="stAlert"][kind="success"] p {
    color: #14532d !important;
}

/* Metrics and captions */
[data-testid="stMetricValue"] {
    font-size: 30px !important;
    color: #123d29 !important;
}

[data-testid="stCaptionContainer"] p {
    font-size: 16px !important;
    color: #294d39 !important;
}

/* Rainfall table */
[data-testid="stDataFrame"] {
    font-size: 17px !important;
}

</style>
""", unsafe_allow_html=True)


# 3. LOAD SAVED MODELS
# =========================================================
@st.cache_resource
def load_mobilenet():
    model = tf.keras.models.load_model(
        str(MOBILENET_PATH), compile=False
    )
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "Model output count does not match the class-name mapping."
        )
    return model, class_names


@st.cache_resource
def load_yolo():
    return YOLO(str(YOLO_PATH))


@st.cache_resource
def load_leaf_validator():
    if not VALIDATOR_PATH.exists():
        raise FileNotFoundError(
            f"Leaf validator not found: {VALIDATOR_PATH}"
        )

    if not VALIDATOR_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Validator configuration not found: {VALIDATOR_CONFIG_PATH}"
        )

    with open(VALIDATOR_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = tf.keras.models.load_model(
        str(VALIDATOR_PATH),
        compile=False
    )

    return model, config


def validate_plant_leaf(image):
    model, config = load_leaf_validator()

    image_size = int(config.get("image_size", 160))
    threshold = float(config.get("leaf_threshold", 0.80))

    image_array = np.asarray(
        image.convert("RGB").resize((image_size, image_size)),
        dtype=np.float32
    )

    batch = np.expand_dims(image_array, axis=0)

    probability = float(
        model.predict(batch, verbose=0)[0][0]
    )

    return probability, threshold


def readable_class(raw_name):
    if "___" in raw_name:
        crop, disease = raw_name.split("___", 1)
        return crop.replace("_", " "), disease.replace("_", " ")
    return raw_name.replace("_", " ")


def classify_leaf(image):
    model, class_names = load_mobilenet()

    # Must match MobileNetV2 training preprocessing.
    image_array = np.asarray(
        image.convert("RGB").resize((224, 224)),
        dtype=np.float32,
    )
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )
    batch = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    ranked_indices = np.argsort(probabilities)[::-1][:3]

    return (
        class_names[best_index],
        float(probabilities[best_index]),
        [
            (class_names[int(i)], float(probabilities[int(i)]))
            for i in ranked_indices
        ],
    )


# =========================================================

# AGROVISION DATE INPUT FORCE FIX
st.markdown("""
<style>

/* AGROVISION DATE INPUT FORCE FIX */

div[data-testid="stDateInput"] div[data-baseweb="input"],
div[data-testid="stDateInput"] div[data-baseweb="input"] > div {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 2px solid #176b3d !important;
    border-radius: 8px !important;
}

div[data-testid="stDateInput"] input,
div[data-testid="stDateInput"] input[type="date"],
div[data-baseweb="datepicker"] input {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #123d29 !important;
    -webkit-text-fill-color: #123d29 !important;
    color-scheme: light !important;
    font-size: 20px !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

div[data-testid="stDateInput"] input::-webkit-datetime-edit,
div[data-testid="stDateInput"] input::-webkit-datetime-edit-fields-wrapper,
div[data-testid="stDateInput"] input::-webkit-datetime-edit-text,
div[data-testid="stDateInput"] input::-webkit-datetime-edit-month-field,
div[data-testid="stDateInput"] input::-webkit-datetime-edit-day-field,
div[data-testid="stDateInput"] input::-webkit-datetime-edit-year-field {
    color: #123d29 !important;
    -webkit-text-fill-color: #123d29 !important;
    opacity: 1 !important;
}

div[data-testid="stDateInput"] button {
    background: #ffffff !important;
    color: #176b3d !important;
}

div[data-testid="stDateInput"] button svg {
    color: #176b3d !important;
    fill: #176b3d !important;
}

</style>
""", unsafe_allow_html=True)


# AGROVISION FULLSCREEN AND LARGE FONT FIX
st.markdown("""
<style>

/* AGROVISION FULLSCREEN AND LARGE FONT FIX */

/* Use almost the entire desktop screen */
.stAppViewContainer .main .block-container,
[data-testid="stMainBlockContainer"],
.main .block-container {
    width: 96vw !important;
    max-width: 1700px !important;
    min-width: 0 !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1.5rem !important;
}

/* Large readable normal text */
.stApp p,
.stApp li,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stCaptionContainer"] {
    font-size: 20px !important;
    line-height: 1.55 !important;
}

/* Headings */
.stApp h1 {
    font-size: 42px !important;
    line-height: 1.2 !important;
}

.stApp h2 {
    font-size: 32px !important;
    line-height: 1.3 !important;
}

.stApp h3 {
    font-size: 27px !important;
    line-height: 1.35 !important;
}

/* AgroVision header */
.brand {
    width: 100% !important;
    max-width: 1600px !important;
}

.brand h1 {
    font-size: 42px !important;
    color: #ffffff !important;
}

.brand p {
    font-size: 19px !important;
    color: #ffffff !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 19px !important;
    font-weight: 700 !important;
    padding: 14px 18px !important;
}

/* Upload labels */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p {
    font-size: 19px !important;
    font-weight: 700 !important;
}

/* Dropdowns and date labels */
[data-testid="stWidgetLabel"] p {
    font-size: 19px !important;
    font-weight: 700 !important;
}

/* Dropdown values */
[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"] {
    font-size: 18px !important;
}

/* Buttons */
.stButton button {
    min-height: 54px !important;
    font-size: 19px !important;
    font-weight: 800 !important;
}

.stButton button p {
    font-size: 19px !important;
}

/* Metrics */
[data-testid="stMetricLabel"] {
    font-size: 19px !important;
}

[data-testid="stMetricValue"] {
    font-size: 34px !important;
}

/* Forecast table */
[data-testid="stDataFrame"] {
    font-size: 18px !important;
}

/* Supporting captions */
[data-testid="stCaptionContainer"] p {
    font-size: 17px !important;
}

/* Keep the existing green header and button contrast */
.brand h1,
.brand p,
.stButton button,
.stButton button p {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# 4. HEADER
# =========================================================
st.markdown("""
<div class="brand">
    <h1>🌱 AgroVision</h1>
    <p>AI-Based Decision Support System</p>
    <p>🌿 Crop Health &nbsp; | &nbsp; 🌧️ Rainfall Forecast
       &nbsp; | &nbsp; 🚜 Smarter Farming Decisions</p>
</div>
""", unsafe_allow_html=True)

# The order of these tabs is intentional.
disease_tab, rainfall_tab = st.tabs([
    "🌿 1. Plant Disease Detection",
    "🌧️ 2. Rainfall Prediction",
])


# =========================================================

# =========================================================
# 5. PLANT DISEASE DETECTION — FIRST SECTION
# =========================================================
with disease_tab:
    st.markdown(
        '<div class="section-title">🌿 Plant Disease Detection</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Upload one leaf image. A single analysis runs MobileNetV2 "
        "classification and YOLO affected-region localization."
    )

    uploaded_file = st.file_uploader(
        "📷 Upload a clear plant leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload",
        help="Use a clear, well-lit image with the leaf visible.",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as error:
            st.error(f"Could not open this image: {error}")
            image = None

        if image is not None:
            st.image(
                image,
                caption="Uploaded leaf image",
                width=480,
            )

            analyze = st.button(
                "🔍 Analyze Leaf — Disease + YOLO",
                type="primary",
                width="stretch",
                key="analyze_leaf_all",
            )

            if analyze:
                try:
                    with st.spinner("Checking whether this is a plant leaf..."):
                        leaf_probability, leaf_threshold = validate_plant_leaf(image)

                    st.write(
                        f"**Plant leaf validation:** "
                        f"{leaf_probability * 100:.2f}%"
                    )

                    if leaf_probability < leaf_threshold:
                        st.error(
                            "This image does not meet the plant-leaf "
                            "validation threshold. Please upload a clear "
                            "photograph of a plant leaf."
                        )
                    else:
                        st.success("Plant leaf accepted. Starting disease analysis.")

                        with st.expander("Leaf validation details"):
                            st.write(
                                f"Acceptance threshold: "
                                f"{leaf_threshold * 100:.0f}%"
                            )

                        result_col1, result_col2 = st.columns(2, gap="large")
    
                        # MobileNetV2 classification
                        with result_col1:
                            st.subheader("🧠 Disease Classification")
                            if not MOBILENET_PATH.exists():
                                st.error(f"MobileNetV2 model not found: {MOBILENET_PATH}")
                            elif not CLASS_NAMES_PATH.exists():
                                st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
                            else:
                                try:
                                    with st.spinner("Classifying the leaf..."):
                                        predicted_class, confidence, top_predictions = (
                                            classify_leaf(image)
                                        )
    
                                    crop, disease = readable_class(predicted_class)
                                    st.success("Classification completed")
                                    st.metric("Predicted crop", crop.title())
                                    st.markdown(
                                        f"**Predicted condition:** {disease.title()}"
                                    )
                                    st.progress(max(0.0, min(confidence, 1.0)))
                                    st.write(
                                        f"**Model confidence:** {confidence * 100:.2f}%"
                                    )
    
                                    if confidence < 0.60:
                                        st.warning(
                                            "Confidence is relatively low. "
                                            "Try another clear image and seek expert confirmation."
                                        )
    
                                    with st.expander("View top 3 predictions"):
                                        for name, score in top_predictions:
                                            crop_name, disease_name = readable_class(name)
                                            st.write(
                                                f"**{crop_name.title()} — "
                                                f"{disease_name.title()}**: {score * 100:.2f}%"
                                            )
    
                                except Exception as error:
                                    st.error(f"Classification failed: {error}")
    
                                               # YOLO temporarily disabled for deployment testing.
                        with result_col2:
                            st.subheader("🎯 Affected Region Localization")
                            st.info(
                                "YOLO localization is temporarily disabled "
                                "while testing the disease classification module."
                            )
    
                except Exception as error:
                    st.error(f"Plant leaf validation failed: {error}")

            st.caption(
                "AI results are decision-support information, not a substitute "
                "for confirmation by an agricultural expert."
            )

    else:
        st.info(
            "Upload a leaf image, then click Analyze Leaf once to run "
            "both MobileNetV2 and YOLO."
        )




# =========================================================
# 6. LIVE RAINFALL FORECAST — SECOND SECTION
# =========================================================


@st.cache_data
def load_rainfall_locations():

    coordinate_file = (
        BASE / "rainfall" / "trained_model"
        / "ap_ts_mandal_coordinates.csv"
    )

    if not coordinate_file.exists():
        raise FileNotFoundError(
            f"Mandal coordinate file not found: {coordinate_file}"
        )

    locations = pd.read_csv(coordinate_file)

    required_columns = [
        "state",
        "district",
        "mandal",
        "latitude",
        "longitude"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in locations.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing coordinate columns: {missing_columns}"
        )

    locations = locations.dropna(
        subset=required_columns
    ).copy()

    locations["state"] = locations["state"].astype(str).str.strip()
    locations["district"] = locations["district"].astype(str).str.strip()
    locations["mandal"] = locations["mandal"].astype(str).str.strip()

    locations["latitude"] = pd.to_numeric(
        locations["latitude"], errors="coerce"
    )

    locations["longitude"] = pd.to_numeric(
        locations["longitude"], errors="coerce"
    )

    locations = locations.dropna(
        subset=["latitude", "longitude"]
    )

    locations = locations[
        locations["state"].str.lower().isin(
            ["andhra pradesh", "telangana"]
        )
    ].copy()

    # Preserve compatibility with the existing dropdown interface.
    locations["tehsil"] = locations["mandal"]

    return locations.sort_values(
        ["state", "district", "mandal"]
    ).reset_index(drop=True)


@st.cache_data(ttl=2592000)
def get_taluka_coordinates(taluka, district, state):

    locations = load_rainfall_locations()

    selected = locations[
        (locations["state"].str.lower() == str(state).strip().lower())
        & (
            locations["district"].str.lower()
            == str(district).strip().lower()
        )
        & (
            locations["mandal"].str.lower()
            == str(taluka).strip().lower()
        )
    ]

    if len(selected) == 0:
        raise ValueError(
            f"No coordinate found for {taluka}, {district}, {state}."
        )

    if len(selected) > 1:
        raise ValueError(
            f"Multiple coordinate records found for "
            f"{taluka}, {district}, {state}."
        )

    latitude = float(selected.iloc[0]["latitude"])
    longitude = float(selected.iloc[0]["longitude"])

    return latitude, longitude


def get_live_rainfall_forecast(latitude, longitude):

    # -----------------------------------------------------
    # API 1: Standard forecast — first 16 days
    # -----------------------------------------------------
    standard_url = "https://api.open-meteo.com/v1/forecast"

    standard_params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "precipitation_sum",
            "precipitation_probability_max",
        ],
        "forecast_days": 16,
        "timezone": "Asia/Kolkata",
    }

    standard_response = requests.get(
        standard_url,
        params=standard_params,
        timeout=60,
    )
    standard_response.raise_for_status()

    standard_data = standard_response.json()
    standard_daily = standard_data.get("daily", {})

    standard_df = pd.DataFrame({
        "Date": standard_daily.get("time", []),
        "Rainfall (mm)": standard_daily.get(
            "precipitation_sum", []
        ),
        "Rain Probability (%)": standard_daily.get(
            "precipitation_probability_max", []
        ),
    })

    standard_df["Date"] = pd.to_datetime(
        standard_df["Date"]
    ).dt.date

    # -----------------------------------------------------
    # API 2: GFS ensemble — up to 35 days
    # -----------------------------------------------------
    ensemble_url = "https://ensemble-api.open-meteo.com/v1/ensemble"

    ensemble_params = {
        "latitude": latitude,
        "longitude": longitude,
        "models": "ncep_gefs05",
        "hourly": "precipitation",
        "forecast_days": 35,
        "timezone": "Asia/Kolkata",
    }

    ensemble_response = requests.get(
        ensemble_url,
        params=ensemble_params,
        timeout=120,
    )
    ensemble_response.raise_for_status()

    ensemble_data = ensemble_response.json()
    hourly = ensemble_data.get("hourly", {})

    hourly_df = pd.DataFrame(hourly)

    if hourly_df.empty:
        raise ValueError("Ensemble forecast returned no data.")

    hourly_df["time"] = pd.to_datetime(hourly_df["time"])
    hourly_df["Date"] = hourly_df["time"].dt.date

    member_columns = [
        column for column in hourly_df.columns
        if column.startswith("precipitation_member")
    ]

    if not member_columns:
        raise ValueError(
            "No ensemble member rainfall values were returned."
        )

    daily_members = (
        hourly_df.groupby("Date")[member_columns]
        .sum(min_count=1)
    )

    valid_members = daily_members.notna().sum(axis=1)

    ensemble_df = pd.DataFrame({
        "Date": daily_members.index,
        "Rainfall (mm)": daily_members.mean(axis=1),
        "Rain Probability (%)": (
            daily_members.ge(0.1).sum(axis=1)
            .div(valid_members.replace(0, np.nan))
            * 100
        ),
    })

    # -----------------------------------------------------
    # Combine the two forecast ranges.
    # Standard forecast for days 1–16.
    # Ensemble forecast for days 17–35.
    # -----------------------------------------------------
    standard_dates = set(standard_df["Date"])

    ensemble_df["Forecast Source"] = "GFS Ensemble"
    standard_df["Forecast Source"] = "Standard Forecast"

    combined = pd.concat(
        [
            standard_df,
            ensemble_df[
                ~ensemble_df["Date"].isin(standard_dates)
            ],
        ],
        ignore_index=True,
    )

    combined = combined.sort_values("Date").reset_index(drop=True)

    combined["Rainfall (mm)"] = pd.to_numeric(
        combined["Rainfall (mm)"], errors="coerce"
    )

    combined["Rain Probability (%)"] = pd.to_numeric(
        combined["Rain Probability (%)"], errors="coerce"
    )

    return combined, len(member_columns)


# ---------------------------------------------------------
# Rainfall user interface
# ---------------------------------------------------------

with rainfall_tab:
    st.markdown(
        '<div class="section-title">🌧️ Live Rainfall Forecast</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Select a state, district, taluka and starting date to view "
        "live weather-model rainfall predictions for seven consecutive days."
    )

    try:
        locations_df = load_rainfall_locations()

        if locations_df.empty:
            st.error("No supported locations were found.")
        else:

            states = sorted(
                locations_df["state"].dropna().unique().tolist()
            )

            selected_state = st.selectbox(
                "🗺️ Select State",
                states,
                key="rain_state",
            )

            state_df = locations_df[
                locations_df["state"] == selected_state
            ]

            districts = sorted(
                state_df["district"].dropna().unique().tolist()
            )

            selected_district = st.selectbox(
                "📍 Select District",
                districts,
                key="rain_district",
            )

            district_df = state_df[
                state_df["district"] == selected_district
            ]

            talukas = sorted(
                district_df["tehsil"].dropna().unique().tolist()
            )

            selected_taluka = st.selectbox(
                "🌾 Select Taluka",
                talukas,
                key="rain_taluka",
            )

            today = pd.Timestamp.now(
                tz="Asia/Kolkata"
            ).date()

            latest_start_date = today + timedelta(days=28)

            starting_date = st.date_input(
                "📅 Select Starting Date",
                value=today,
                min_value=today,
                max_value=latest_start_date,
                key="rain_start_date",
                help=(
                    "Select a date within the available 35-day forecast "
                    "window. Seven complete forecast days must be available."
                ),
            )

            predict_rainfall = st.button(
                "🌧️ Predict Rainfall for 7 Days",
                type="primary",
                use_container_width=True,
                key="estimate_rainfall",
            )

            if predict_rainfall:

                with st.spinner(
                    "Finding location and retrieving live weather forecasts..."
                ):

                    latitude, longitude = (
                        get_taluka_coordinates(
                            selected_taluka,
                            selected_district,
                            selected_state,
                        )
                    )

                    forecast_df, member_count = (
                        get_live_rainfall_forecast(
                            latitude,
                            longitude,
                        )
                    )

                forecast_df = forecast_df.sort_values(
                    "Date"
                ).reset_index(drop=True)

                forecast_end = forecast_df["Date"].max()

                requested_end = starting_date + timedelta(days=6)

                if (
                    starting_date < forecast_df["Date"].min()
                    or requested_end > forecast_end
                ):
                    st.error(
                        "The selected date does not have seven complete "
                        "forecast days available. Please select an earlier date."
                    )
                else:

                    result_df = forecast_df[
                        (forecast_df["Date"] >= starting_date)
                        & (forecast_df["Date"] <= requested_end)
                    ].copy()

                    if len(result_df) != 7:
                        st.error(
                            "The weather API did not return all seven requested days."
                        )
                    else:

                        # Status considers both rainfall amount and probability.
                        def classify_rainfall(row):
                            rainfall = row["Rainfall (mm)"]
                            probability = row["Rain Probability (%)"]

                            if pd.isna(rainfall) or pd.isna(probability):
                                return "Unavailable"
                            if rainfall <= 0:
                                return "No Rain Expected"
                            if probability >= 40:
                                return "Rain Expected"
                            return "Low Rain Chance"

                        result_df["Rainfall Status"] = result_df.apply(
                            classify_rainfall, axis=1
                        )

                        st.subheader(
                            f"📍 {selected_taluka}, "
                            f"{selected_district}, {selected_state}"
                        )

                        st.caption(
                            f"Weather location: {selected_taluka}, {selected_district} | "
                            f"Coordinates: {latitude:.4f}, {longitude:.4f}"
                        )

                        st.write(
                            f"Forecast period: "
                            f"{starting_date:%d %b %Y} to "
                            f"{requested_end:%d %b %Y}"
                        )

                        standard_days = (
                            result_df["Forecast Source"]
                            == "Standard Forecast"
                        ).sum()

                        ensemble_days = (
                            result_df["Forecast Source"]
                            == "GFS Ensemble"
                        ).sum()

                        st.caption(
                            f"Forecast sources: {standard_days} standard "
                            f"forecast days and {ensemble_days} ensemble "
                            f"guidance days. The ensemble calculation uses "
                            f"{member_count} members."
                        )

                        total = result_df["Rainfall (mm)"].sum()
                        average = result_df["Rainfall (mm)"].mean()

                        metric1, metric2 = st.columns(2)

                        metric1.metric(
                            "7-day estimated rainfall",
                            f"{total:.2f} mm",
                        )

                        metric2.metric(
                            "Average daily rainfall",
                            f"{average:.2f} mm/day",
                        )

                        display_df = result_df.copy()

                        display_df["Date"] = display_df["Date"].map(
                            lambda value: value.strftime("%d %b %Y")
                        )

                        display_df["Rainfall (mm)"] = (
                            display_df["Rainfall (mm)"].map(
                                lambda value:
                                f"{value:.2f} mm"
                                if pd.notna(value)
                                else "Unavailable"
                            )
                        )

                        display_df["Rain Probability (%)"] = (
                            display_df["Rain Probability (%)"].map(
                                lambda value:
                                f"{value:.1f}%"
                                if pd.notna(value)
                                else "Unavailable"
                            )
                        )

                        st.subheader("📋 Daily Rainfall Forecast")

                        st.dataframe(
                            display_df[
                                [
                                    "Date",
                                    "Rainfall (mm)",
                                    "Rainfall Status",
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

                        chart_df = result_df.set_index("Date")[
                            ["Rainfall (mm)"]
                        ]

                        st.subheader("📊 Seven-Day Rainfall Forecast")
                        st.bar_chart(chart_df)

                        st.info(
                            "Rainfall status is based on a 40% rain-probability "
                            "threshold. The standard forecast is used for the "
                            "first 16 days; extended dates use GFS ensemble "
                            "guidance. Long-range forecasts are uncertain and "
                            "should be treated as decision-support information, "
                            "not guaranteed rainfall predictions."
                        )

    except Exception as error:
        st.error(
            f"Could not load or calculate live rainfall forecasts: {error}"
        )

# 7. FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    🌱 AgroVision | AI-Based Decision Support System
    <br>Healthy Crops • Smarter Decisions • Sustainable Agriculture
</div>
""", unsafe_allow_html=True)
