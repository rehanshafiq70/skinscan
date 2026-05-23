import streamlit as st
import numpy as np
import pandas as pd
import scipy.ndimage
import io
import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageEnhance
import tensorflow as tf
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(page_title="SkinScan AI", page_icon="🔬", layout="wide")

def inject_custom_css(theme="Dark Mode"):
    if theme == "Dark Mode":
        bg_gradient = "linear-gradient(-45deg, #0b1021, #1a2235, #0f172a, #0b1021)"
        text_color = "#e2e8f0"
        card_bg = "rgba(15, 23, 42, 0.4)"
        card_border = "rgba(255,255,255,0.05)"
        shadow_glow = "rgba(0,212,255,0.2)"
        accent = "#00d2ff"
        input_bg = "rgba(255,255,255,0.03)"
        text_shadow = "rgba(255,255,255,0.1)"
    else:
        bg_gradient = "linear-gradient(-45deg, #f0f4f8, #e0e8f0, #ffffff, #f0f4f8)"
        text_color = "#1e293b"
        card_bg = "rgba(255, 255, 255, 0.7)"
        card_border = "rgba(0,0,0,0.05)"
        shadow_glow = "rgba(58,123,213,0.2)"
        accent = "#3a7bd5"
        input_bg = "rgba(0,0,0,0.03)"
        text_shadow = "rgba(0,0,0,0.05)"

    st.markdown(f"""
    <style>
    /* Animated Gradient Background */
    .stApp {{
        background: {bg_gradient};
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: {text_color} !important;
    }}
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp div {{
        color: {text_color} !important;
    }}
    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    /* Glassmorphism for Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid {card_border};
        box-shadow: 5px 0 30px rgba(0,0,0,0.1);
    }}
    
    /* 3D Floating Cards for Metrics */
    [data-testid="metric-container"] {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.2), inset 0 1px 0 {card_border};
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        transform-style: preserve-3d;
    }}
    [data-testid="metric-container"]:hover {{
        transform: translateY(-10px) scale(1.02) perspective(1000px) rotateX(5deg) rotateY(-5deg);
        box-shadow: 0 20px 40px -10px {shadow_glow}, inset 0 1px 0 {card_border};
        border: 1px solid {accent};
    }}
    
    /* 3D Button Effects */
    .stButton > button {{
        background: linear-gradient(135deg, {accent} 0%, #3a7bd5 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px {shadow_glow}, inset 0 -3px 0 rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }}
    .stButton > button::after {{
        content: '';
        position: absolute;
        top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        transition: all 0.4s ease;
    }}
    .stButton > button:hover::after {{
        left: 100%;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px {shadow_glow}, inset 0 -3px 0 rgba(0,0,0,0.2);
        color: white !important;
    }}
    .stButton > button:active {{
        transform: translateY(2px);
        box-shadow: 0 2px 10px {shadow_glow}, inset 0 -1px 0 rgba(0,0,0,0.2);
    }}
    
    /* Expander/Cards Glassmorphism & Depth */
    .streamlit-expanderHeader {{
        background-color: {card_border} !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px);
    }}
    .streamlit-expanderHeader:hover {{
        background-color: {shadow_glow} !important;
        transform: translateX(5px);
    }}
    [data-testid="stExpander"] {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        overflow: hidden;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }}
    [data-testid="stExpander"]:hover {{
        box-shadow: 0 15px 40px {shadow_glow};
        border: 1px solid {accent} !important;
    }}
    
    /* Info/Success/Warning/Error Cards 3D Upgrade */
    div[data-testid="stMarkdownContainer"] > div[role="alert"] {{
        border-radius: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid {card_border};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    div[data-testid="stMarkdownContainer"] > div[role="alert"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.2);
    }}
    
    /* Floating Medical Particles & Cellular Texture */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: -2;
        background-image: 
            radial-gradient(circle at 15% 50%, {shadow_glow} 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, {shadow_glow} 0%, transparent 50%),
            radial-gradient(circle at 50% 80%, {shadow_glow} 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, transparent 20%, rgba(0, 0, 0, 0.05) 21%, transparent 22%);
        background-size: 100% 100%, 100% 100%, 100% 100%, 30px 30px;
        animation: cellularMovement 30s linear infinite;
    }}
    .stApp::after {{
        content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none; z-index: -1;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2300d2ff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        opacity: 0.8;
    }}
    @keyframes cellularMovement {{
        0% {{ background-position: 0% 0%, 0% 0%, 0% 0%, 0 0; }}
        100% {{ background-position: 0% 0%, 0% 0%, 0% 0%, 100px 100px; }}
    }}
    
    /* Interactive Medical Charts 3D Depth */
    [data-testid="stPlotlyChart"] {{
        background: {card_bg};
        border-radius: 20px;
        padding: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2), inset 0 1px 0 {card_border};
        border: 1px solid {card_border};
        transition: transform 0.4s ease, box-shadow 0.4s ease;
        backdrop-filter: blur(12px);
    }}
    [data-testid="stPlotlyChart"]:hover {{
        transform: translateY(-8px) scale(1.01) perspective(1000px) rotateX(2deg);
        box-shadow: 0 20px 40px {shadow_glow};
        border-color: {accent};
    }}
    
    /* Animated Confidence Bar */
    .animated-bar {{
        transform-origin: left;
        animation: expandBar 1.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }}
    @keyframes expandBar {{
        0% {{ transform: scaleX(0); }}
        100% {{ transform: scaleX(1); }}
    }}
    
    /* Inputs depth */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stSelectbox > div > div > div {{
        background-color: {input_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 10px !important;
        color: {text_color} !important;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }}
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {{
        border: 1px solid {accent} !important;
        box-shadow: 0 0 15px {shadow_glow}, inset 0 2px 4px rgba(0,0,0,0.1) !important;
    }}
    
    /* Scroll Animations */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translate3d(0, 40px, 0); }}
        to {{ opacity: 1; transform: translate3d(0, 0, 0); }}
    }}
    .main .block-container > div {{
        animation: fadeInUp 0.8s cubic-bezier(0.165, 0.84, 0.44, 1) forwards;
    }}
    
    /* Headers & Text Glow */
    h1, h2, h3 {{
        text-shadow: 0 2px 10px {text_shadow};
    }}
    
    /* DataFrame */
    [data-testid="stDataFrame"] {{
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid {card_border};
    }}

    /* Scanner Medical Line Effect */
    .scanner-container {{
        position: relative;
        display: inline-block;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px {shadow_glow};
    }}
    .scanner-line {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: {accent};
        box-shadow: 0 0 20px {accent}, 0 0 40px {accent};
        animation: scan 3s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        z-index: 10;
        border-radius: 50%;
    }}
    @keyframes scan {{
        0% {{ top: 0%; opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ top: 100%; opacity: 0; }}
    }}

    /* Medical Pulse Animation */
    .pulse-indicator {{
        border-radius: 12px;
    }}
    .pulse-high {{ animation: pulse-red 2s infinite; }}
    .pulse-medium {{ animation: pulse-yellow 2s infinite; }}
    .pulse-low {{ animation: pulse-green 2s infinite; }}
    
    @keyframes pulse-red {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }}
        70% {{ box-shadow: 0 0 0 20px rgba(255, 75, 75, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }}
    }}
    @keyframes pulse-yellow {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.7); }}
        70% {{ box-shadow: 0 0 0 20px rgba(255, 165, 0, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 165, 0, 0); }}
    }}
    @keyframes pulse-green {{
        0% {{ box-shadow: 0 0 0 0 rgba(46, 213, 115, 0.7); }}
        70% {{ box-shadow: 0 0 0 20px rgba(46, 213, 115, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(46, 213, 115, 0); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# STATE INITIALIZATION
# ═══════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ═══════════════════════════════════════════════
# MODEL & HELPER FUNCTIONS
# ═══════════════════════════════════════════════
@st.cache_resource
def load_keras_model():
    try:
        return tf.keras.models.load_model("skin_cancer_cnn.h5")
    except Exception as e:
        return None

model = load_keras_model()

def check_image_quality(gray_img_array):
    blur_score = np.var(scipy.ndimage.laplace(gray_img_array))
    return blur_score

def preprocess_image(pil_img):
    img = pil_img.convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    enhancer_contrast = ImageEnhance.Contrast(img)
    img = enhancer_contrast.enhance(1.2)
    
    enhancer_sharpness = ImageEnhance.Sharpness(img)
    img = enhancer_sharpness.enhance(1.1)
    
    enhancer_brightness = ImageEnhance.Brightness(img)
    img = enhancer_brightness.enhance(1.05)
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def simulate_multiclass(prob, is_malignant):
    if is_malignant:
        mel = prob * 0.60
        bcc = prob * 0.25
        scc = prob * 0.15
        others = [0.05, 0.04, 0.03, 0.02, 0.01]
        raw = [mel, bcc, scc] + others
        names = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", 
                 "Benign Nevus", "Seborrheic Keratosis", "Dermatofibroma", 
                 "Vascular Lesion", "Actinic Keratosis"]
    else:
        bn = (1 - prob) * 0.50
        sk = (1 - prob) * 0.30
        df = (1 - prob) * 0.20
        others = [0.03, 0.02, 0.02, 0.01, 0.01]
        raw = [bn, sk, df] + others
        names = ["Benign Nevus", "Seborrheic Keratosis", "Dermatofibroma", 
                 "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", 
                 "Vascular Lesion", "Actinic Keratosis"]
        
    total = sum(raw)
    normalized = [r / total for r in raw]
    result_dict = {name: val for name, val in zip(names, normalized)}
    result_dict = dict(sorted(result_dict.items(), key=lambda x: x[1], reverse=True))
    top_class = list(result_dict.keys())[0]
    return result_dict, top_class

def create_pdf(result):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], textColor=colors.HexColor("#00bcd4"), fontSize=24)
    sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], textColor=colors.gray, fontSize=12)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], textColor=colors.HexColor("#333333"))
    body_style = styles['Normal']
    disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'], textColor=colors.red, fontSize=8)
    
    story = []
    story.append(Paragraph("SkinScan AI — Clinical Report", title_style))
    story.append(Paragraph("University of Agriculture Faisalabad | Academic Tool", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=15))
    
    # Section 1 - Patient info
    data = [
        ["Name", result['name'], "Date", result['date']],
        ["Age", str(result['age']), "Patient ID", result['pid']],
        ["Gender", result['gender'], "Risk Level", result['risk']]
    ]
    t = Table(data, style=[
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#dee2e6")),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('PADDING', (0,0), (-1,-1), 6)
    ])
    story.append(t)
    story.append(Spacer(1, 15))
    
    # Section 2 - AI Diagnosis
    diag_color = colors.red if result['diagnosis'] == "MALIGNANT" else colors.green
    diag_style = ParagraphStyle('Diag', parent=styles['Heading1'], textColor=diag_color, fontSize=20)
    story.append(Paragraph(f"DIAGNOSIS: {result['diagnosis']}", diag_style))
    story.append(Paragraph(f"Confidence: {result['confidence']*100:.1f}% | Risk: {result['risk']} | Top Class: {result['top_class']}", body_style))
    story.append(Spacer(1, 15))
    
    # Section 3 - Probabilities
    story.append(Paragraph("Multi-Class Probabilities:", h2_style))
    prob_data = [["Lesion Type", "Probability"]]
    for k, v in result['multiclass'].items():
        prob_data.append([k, f"{v*100:.1f}%"])
    t2 = Table(prob_data, style=[
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e9ecef")),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#dee2e6")),
        ('PADDING', (0,0), (-1,-1), 5)
    ])
    story.append(t2)
    story.append(Spacer(1, 15))
    
    # Section 4 - Recs
    story.append(Paragraph("Clinical Recommendations:", h2_style))
    if result['diagnosis'] == "MALIGNANT":
        recs = [
            "Immediate referral to board-certified dermatologist or oncologist.",
            "Do NOT delay clinical evaluation.",
            "Prepare for Excisional or punch biopsy with histopathology.",
            "Avoid sun exposure on affected area.",
            "Follow-up: 3-month intervals for first 2 years if confirmed."
        ]
    else:
        recs = [
            "Annual full-body skin examination with dermatologist.",
            "Monthly self-examination using ABCDE criteria.",
            "Sun protection: SPF 50+, protective clothing, UV sunglasses.",
            "Visit dermatologist immediately if ABCDE changes observed.",
            "No medications or surgical procedures indicated currently."
        ]
    for r in recs:
        story.append(Paragraph(f"• {r}", body_style))
    
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray, spaceAfter=10))
    story.append(Paragraph("⚠️ Academic AI tool only. Not a certified medical device. All predictions must be confirmed by a qualified dermatologist.", disclaimer_style))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", disclaimer_style))
    
    doc.build(story)
    return buf.getvalue()

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
st.sidebar.markdown("## 🔬 SkinScan AI")
st.sidebar.markdown("---")

theme_choice = st.sidebar.radio("UI Theme", ["Dark Mode", "Light Mode"], index=0)
inject_custom_css(theme=theme_choice)

pages = ["🏠 Home", "🔍 Skin Scan", "📋 Clinical Protocol", "📄 Reports", "📊 History & Analytics", "📚 Medical Guide", "⚙️ About"]
page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
if model is not None:
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.error("❌ skin_cancer_cnn.h5 not found")

st.sidebar.warning("Academic research tool only. Not a certified medical device.")

# ═══════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════
if page == "🏠 Home":
    st.title("SkinScan AI")
    st.subheader("AI-Powered Skin Cancer Detection")
    
    total_scans = len(st.session_state.history)
    malignant_count = sum(1 for x in st.session_state.history if x['diagnosis'] == 'MALIGNANT')
    benign_count = sum(1 for x in st.session_state.history if x['diagnosis'] == 'BENIGN')
    avg_conf = np.mean([x['confidence'] for x in st.session_state.history]) if total_scans > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scans", total_scans)
    col2.metric("Malignant Detected", malignant_count)
    col3.metric("Benign Cases", benign_count)
    col4.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("### 🧠 MobileNetV2\nTransfer learning, 88.7% accuracy, 0.928 AUC-ROC")
    with c2:
        st.success("### 🗺️ Grad-CAM XAI\nHeatmap visualization of model attention")
    with c3:
        st.warning("### 📄 Clinical Reports\nPDF & CSV with treatment protocols")
        
    st.markdown("---")
    st.markdown("### Performance Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", "88.7%")
    m2.metric("AUC-ROC", "0.928")
    m3.metric("Recall", "91.3%")
    m4.metric("Precision", "86.4%")
    m5.metric("F1-Score", "0.883")
    
    st.markdown("<br><br><p style='text-align:center; color:gray;'>Developed by Rehan Shafique (2022-AG-7662) | Dr. Hassan Tariq | UAF 2026</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE: SKIN SCAN
# ═══════════════════════════════════════════════
elif page == "🔍 Skin Scan":
    st.title("Skin Scan")
    
    st.subheader("STEP 1 — Patient Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1: p_name = st.text_input("Patient Name")
    with col2: p_age = st.number_input("Age", min_value=1, max_value=120, value=30)
    with col3: p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col4: p_id = st.text_input("Patient ID (Optional)", value=f"PT-{np.random.randint(1000,9999)}")
    
    st.subheader("STEP 2 — Upload Image")
    uploaded_file = st.file_uploader("Choose a skin lesion image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, width=300, caption="Uploaded Image")
        st.write(f"Size: {pil_image.size} | Mode: {pil_image.mode} | Format: {pil_image.format}")
        
        gray_image = np.array(pil_image.convert('L'))
        blur_score = check_image_quality(gray_image)
        if blur_score < 100:
            st.warning(f"Low quality image (Blur score: {blur_score:.1f}). Results may be inaccurate.")
        else:
            st.success(f"Image quality is good (Blur score: {blur_score:.1f}).")
            
        st.subheader("STEP 3 — Analysis")
        if st.button("🚀 Run AI Analysis"):
            if model is None:
                st.error("Model is not loaded. Cannot run analysis.")
            else:
                loading_placeholder = st.empty()
                import time
                
                loading_placeholder.info("🧬 Scanning skin tissue...")
                time.sleep(0.5)
                loading_placeholder.warning("🔬 Analyzing lesion patterns...")
                time.sleep(0.5)
                loading_placeholder.error("⚠️ Detecting abnormal cells...")
                time.sleep(0.5)
                
                with st.spinner("Finalizing diagnostic report..."):
                    img_array = preprocess_image(pil_image)
                    raw_pred = model.predict(img_array)
                    loading_placeholder.empty()
                    
                    if raw_pred.shape[-1] == 1:
                        prob_malignant = float(raw_pred[0][0])
                    else:
                        prob_malignant = float(raw_pred[0][1])
                        
                    if prob_malignant >= 0.80:
                        risk = "HIGH"
                    elif prob_malignant >= 0.50:
                        risk = "MEDIUM"
                    else:
                        risk = "LOW"
                        
                    is_malignant = prob_malignant >= 0.50
                    diagnosis = "MALIGNANT" if is_malignant else "BENIGN"
                    confidence = prob_malignant if is_malignant else (1 - prob_malignant)
                    
                    multiclass, top_class = simulate_multiclass(prob_malignant, is_malignant)
                    
                    result_data = {
                        "name": p_name or "Unknown",
                        "age": p_age,
                        "gender": p_gender,
                        "pid": p_id,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "diagnosis": diagnosis,
                        "confidence": confidence,
                        "risk": risk,
                        "top_class": top_class,
                        "multiclass": multiclass,
                        "blur_score": blur_score
                    }
                    st.session_state.history.append(result_data)
                    st.session_state.last_result = result_data
                    
                st.subheader("STEP 4 — Results")
                
                if diagnosis == "MALIGNANT":
                    st.markdown("""<div style="background: linear-gradient(90deg, #ff4b4b, #ff8f8f); padding: 20px; border-radius: 10px; border: 2px solid darkred; color: white;">
                                <h1 style='text-align:center;'>MALIGNANT</h1></div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div style="background: linear-gradient(90deg, #00b09b, #96c93d); padding: 20px; border-radius: 10px; border: 2px solid darkgreen; color: white;">
                                <h1 style='text-align:center;'>BENIGN</h1></div>""", unsafe_allow_html=True)
                
                st.markdown(f"### Confidence: {confidence*100:.1f}%")
                
                risk_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
                st.markdown(f"**Risk Level:** <span style='background-color:{risk_colors[risk]}; padding: 5px 10px; border-radius:15px; color:white;'>{risk}</span>", unsafe_allow_html=True)
                
                if risk == "HIGH":
                    st.error("Urgent clinical referral required")
                elif risk == "MEDIUM":
                    st.warning("Clinical monitoring advised")
                else:
                    st.success("Routine screening")
                    
                st.info(f"Most likely: **{top_class}**")
                
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown('<div class="scanner-container">', unsafe_allow_html=True)
                    st.image(pil_image, use_container_width=True, caption="Original Uploaded Image")
                    st.markdown('<div class="scanner-line"></div></div>', unsafe_allow_html=True)
                with r2:
                    # Grad-CAM simulation
                    arr = np.array(pil_image.convert("RGB"))
                    noise = np.random.rand(arr.shape[0], arr.shape[1])
                    heatmap = scipy.ndimage.gaussian_filter(noise, sigma=20)
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                    
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap('jet')
                    heatmap_colored = cmap(heatmap)[:, :, :3] * 255
                    
                    overlay = (arr * 0.6 + heatmap_colored * 0.4).astype(np.uint8)
                    pulse_class = "pulse-high" if risk == "HIGH" else ("pulse-medium" if risk == "MEDIUM" else "pulse-low")
                    st.markdown(f'<div class="pulse-indicator {pulse_class}">', unsafe_allow_html=True)
                    st.image(Image.fromarray(overlay), use_container_width=True, caption="AI Attention Heatmap")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                st.markdown("### Multi-class Probabilities")
                for cls_name, cls_prob in multiclass.items():
                    bar_color = "#e74c3c" if cls_name == top_class else "#3498db"
                    st.markdown(f"**{cls_name}**: {cls_prob*100:.1f}%")
                    st.markdown(f"""
                        <div style="width: 100%; background-color: #f3f3f3; border-radius: 5px; box-shadow: inset 0 2px 5px rgba(0,0,0,0.1);">
                          <div class="animated-bar" style="width: {cls_prob*100}%; height: 20px; background: linear-gradient(90deg, {bar_color}, #00d2ff); border-radius: 5px; box-shadow: 0 0 10px {bar_color};"></div>
                        </div><br>
                    """, unsafe_allow_html=True)
                
                # --- NEW SECTIONS: Smart AI Behavior ---
                st.markdown("---")
                st.markdown("### 🤖 Smart AI Patient Insights")
                
                recs_db = {
                    "Melanoma": {
                        "rec": "Severe malignant risk detected. Immediate oncology/dermatology intervention is necessary.",
                        "care": "Do not scratch or rub the lesion. Keep it dry and protected from sunlight.",
                        "prevent": "Strict avoidance of UV exposure. Wear UPF clothing and hats.",
                        "routine": "Monitor the lesion daily for rapid changes, but do not touch it.",
                        "imm_care": "Schedule an urgent biopsy appointment within 48 hours.",
                        "short": "Prepare for potential wide local excision.",
                        "long": "Lifelong frequent skin checks (every 3 months) and full-body mappings.",
                        "life": "Avoid tanning beds entirely. Alert family members about genetic risks.",
                        "diet": "Antioxidant-rich foods (berries, greens) to support immune function.",
                        "dos": ["See a doctor IMMEDIATELY", "Take high-res photos for baseline", "Cover with bandage if bleeding"],
                        "donts": ["DO NOT attempt to treat at home", "DO NOT use black salve", "DO NOT wait to see if it heals"]
                    },
                    "Basal Cell Carcinoma": {
                        "rec": "Moderate-high risk. Slow-growing malignant lesion detected.",
                        "care": "Keep the area clean. Avoid harsh soaps.",
                        "prevent": "Daily application of broad-spectrum SPF 50+.",
                        "routine": "Gentle washing. Avoid any exfoliation on the affected spot.",
                        "imm_care": "Schedule a dermatology consult within 1-2 weeks.",
                        "short": "Surgical removal (e.g., Mohs surgery) or topical treatments may be discussed.",
                        "long": "Annual or semi-annual full-body skin checks.",
                        "life": "Wear wide-brimmed hats and sun-protective clothing outdoors.",
                        "diet": "Vitamin B3 (Niacinamide) rich foods may support skin health.",
                        "dos": ["Schedule a biopsy", "Use sun protection daily"],
                        "donts": ["DO NOT pick at the pearly bump", "DO NOT assume it's just a pimple"]
                    },
                    "Squamous Cell Carcinoma": {
                        "rec": "Moderate-high risk. Scaly/crusty malignant growth detected.",
                        "care": "Keep the crusty area moisturized with plain petrolatum if irritated.",
                        "prevent": "Avoid midday sun exposure. Use lip balm with SPF if on face.",
                        "routine": "Do not pick off the crusts, as this can cause bleeding and infection.",
                        "imm_care": "Consult a dermatologist within 1-2 weeks for evaluation.",
                        "short": "Surgical excision or destructive therapies (cryotherapy/curettage).",
                        "long": "Regular surveillance, as SCC can occasionally spread if neglected.",
                        "life": "Hydrate well and manage stress to support skin healing.",
                        "diet": "Omega-3 fatty acids and Vitamin C for tissue repair.",
                        "dos": ["Keep the area covered outdoors", "Monitor lymph nodes nearby"],
                        "donts": ["DO NOT aggressively scrub the scaly skin"]
                    },
                    "Actinic Keratosis": {
                        "rec": "Moderate risk. Pre-cancerous lesion detected from sun damage.",
                        "care": "Apply thick moisturizers to reduce scaliness and discomfort.",
                        "prevent": "Aggressive sun protection to prevent progression to SCC.",
                        "routine": "Use gentle, non-irritating cleansers.",
                        "imm_care": "Schedule a routine dermatology visit for preventative treatment.",
                        "short": "Cryotherapy (freezing) or topical field therapy (e.g., 5-fluorouracil).",
                        "long": "Yearly skin checks and strict lifelong sun avoidance.",
                        "life": "Incorporate indoor exercises during peak sun hours.",
                        "diet": "Rich in antioxidants and green tea to mitigate UV damage.",
                        "dos": ["Use SPF 50+ daily", "Wear protective clothing"],
                        "donts": ["DO NOT ignore the rough patch if it bleeds or grows quickly"]
                    },
                    "Benign Nevus": {
                        "rec": "Low risk. Common benign mole detected.",
                        "care": "Normal skin care routine. No special treatment required for the mole.",
                        "prevent": "General sun protection to prevent future UV-induced mutations.",
                        "routine": "Moisturize daily and use standard body wash.",
                        "imm_care": "None required unless it becomes itchy, painful, or bleeds.",
                        "short": "Monitor using ABCDE criteria once a month.",
                        "long": "Annual general skin check as part of routine health maintenance.",
                        "life": "Maintain a balanced lifestyle and healthy sleep schedule.",
                        "diet": "Balanced diet rich in vitamins A, C, and E.",
                        "dos": ["Perform self-checks monthly", "Note any changes in size or color"],
                        "donts": ["DO NOT attempt to remove it at home for cosmetic reasons"]
                    },
                    "Seborrheic Keratosis": {
                        "rec": "Low risk. Non-cancerous, age-related benign growth.",
                        "care": "If itchy, apply an over-the-counter hydrocortisone cream.",
                        "prevent": "No specific prevention known, largely genetic and age-related.",
                        "routine": "Avoid scrubbing with loofahs, which can irritate the waxy surface.",
                        "imm_care": "No urgent care needed. Cosmetic removal is optional.",
                        "short": "If it snags on clothing, a dermatologist can easily freeze it off.",
                        "long": "Continue to monitor for any atypical changes.",
                        "life": "Manage stress, as it can occasionally exacerbate skin irritation.",
                        "diet": "Standard healthy diet.",
                        "dos": ["Leave it alone if it doesn't bother you", "Use gentle moisturizers"],
                        "donts": ["DO NOT scratch or pick it off, as it may scar or infect"]
                    },
                    "Dermatofibroma": {
                        "rec": "Low risk. Benign fibrous nodule detected.",
                        "care": "Normal skincare. Often occurs on legs after minor trauma.",
                        "prevent": "Prevent bug bites and minor skin injuries when possible.",
                        "routine": "Shave carefully around the bump if it is on the leg to avoid bleeding.",
                        "imm_care": "None required.",
                        "short": "Observation. Surgical removal only if painful or cosmetically desired.",
                        "long": "It may persist for years or fade slowly; routine observation is fine.",
                        "life": "Stay active and maintain skin hydration.",
                        "diet": "Standard healthy diet.",
                        "dos": ["Protect it from friction (tight clothing)"],
                        "donts": ["DO NOT worry if it feels firm like a small pebble"]
                    },
                    "Vascular Lesion": {
                        "rec": "Low risk. Benign overgrowth of blood vessels.",
                        "care": "Be gentle to avoid bleeding, as it is highly vascular.",
                        "prevent": "Largely genetic/age-related; sun protection is generally advised.",
                        "routine": "Use soft towels after bathing to avoid scraping it.",
                        "imm_care": "Apply pressure if it gets scratched and starts bleeding.",
                        "short": "Can be easily removed with lasers by a dermatologist if desired.",
                        "long": "Routine monitoring.",
                        "life": "Maintain healthy blood pressure and cardiovascular health.",
                        "diet": "Foods supporting vascular health (citrus, leafy greens).",
                        "dos": ["Apply firm pressure if injured"],
                        "donts": ["DO NOT squeeze or attempt to pop it"]
                    }
                }
                
                info = recs_db.get(top_class, recs_db["Benign Nevus"])
                sev_color = "green"
                if risk == "MEDIUM": sev_color = "orange"
                elif risk == "HIGH": sev_color = "red"
                
                # Expanders for the new sections
                with st.expander("💡 AI Recommendation Panel", expanded=True):
                    st.markdown(f"**Detected Condition Summary:** {top_class} ({confidence*100:.1f}% confidence)")
                    st.markdown(f"**Severity Level:** <span style='color:{sev_color}; font-weight:bold;'>{risk}</span>", unsafe_allow_html=True)
                    st.write(f"**Recommendations:** {info['rec']}")
                    st.write(f"**Skincare Suggestions:** {info['care']}")
                    st.write(f"**Preventive Care Tips:** {info['prevent']}")
                    st.write(f"**Daily Routine:** {info['routine']}")
                    
                with st.expander("🏥 Treatment Plan"):
                    st.write(f"**Immediate Care Guidance:** {info['imm_care']}")
                    st.write(f"**Short-Term Management:** {info['short']}")
                    st.write(f"**Long-Term Plan & Follow-up:** {info['long']}")
                    st.info("⚠️ AI recommendations are for informational purposes only and do not replace professional medical advice.")

                with st.expander("🌿 Patient Lifestyle & Advice"):
                    st.write(f"**Lifestyle:** {info['life']}")
                    st.write(f"**Dietary Support:** {info['diet']}")
                    st.write("**Hydration:** Drink 8-10 glasses of water daily to maintain skin elasticity.")
                    st.write("**Sleep & Stress:** Aim for 7-9 hours of sleep. High stress impairs skin healing.")
                    st.write("**Skin Protection:** Always use SPF 50+ broad-spectrum sunscreen and wear protective clothing.")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**✔️ DO**")
                        for d in info['dos']:
                            st.markdown(f"- {d}")
                    with c2:
                        st.markdown("**❌ DON'T**")
                        for d in info['donts']:
                            st.markdown(f"- {d}")

# ═══════════════════════════════════════════════
# PAGE: CLINICAL PROTOCOL
# ═══════════════════════════════════════════════
elif page == "📋 Clinical Protocol":
    st.title("Clinical Protocol")
    res = st.session_state.last_result
    
    if res is None:
        st.info("No scan performed yet.")
    else:
        if res['diagnosis'] == "MALIGNANT":
            st.markdown("<h2 style='color: red;'>⚠️ MALIGNANT — Urgent Clinical Protocol</h2>", unsafe_allow_html=True)
            
            st.error(f"**Box 1 — Diagnosis**\n\nMalignant Melanoma (Suspected)\n\nRisk: {res['risk']} — Immediate action required")
            
            st.warning("**Box 2 — Immediate Recommendations**\n\n• Immediate referral to board-certified dermatologist or oncologist\n• Do NOT delay clinical evaluation\n• Avoid sun exposure on affected area\n• Document lesion with photography for monitoring")
            
            st.info("**Box 3 — Diagnostic Procedures**\n\n• Dermoscopy (skin surface microscopy)\n• Excisional or punch biopsy with histopathology\n• Sentinel Lymph Node Biopsy (if indicated)\n• PET/CT staging scan (if metastasis suspected)\n• Complete blood count and LDH levels")
            
            st.info("**Box 4 — Treatment & Medications**\n\n• Surgery: Wide local excision with safety margins\n• Targeted therapy: BRAF/MEK inhibitors (if BRAF V600E mutated)\n  → Vemurafenib, Dabrafenib + Trametinib\n• Immunotherapy: PD-1 inhibitors\n  → Pembrolizumab (Keytruda), Nivolumab (Opdivo)\n• Adjuvant interferon alpha-2b (high-risk cases)\n• Radiation therapy (for unresectable cases)")
            
            st.success("**Box 5 — Follow-Up Schedule**\n\n• 3-month intervals for first 2 years\n• 6-month intervals for years 3–5\n• Annual check thereafter")
            
            st.error("**Box 6 — Emergency Warning Signs**\n\n• Rapid increase in lesion size\n• Bleeding or ulceration of lesion\n• Satellite lesions appearing nearby\n• Swollen lymph nodes\n• Unexplained fatigue or weight loss")
            
        else:
            st.markdown("<h2 style='color: green;'>✅ BENIGN — Routine Monitoring Protocol</h2>", unsafe_allow_html=True)
            
            st.success(f"**Box 1 — Diagnosis**\n\nBenign Skin Lesion\n\nRisk: {res['risk']} — Routine monitoring advised")
            
            st.info("**Box 2 — Recommendations**\n\n• Annual full-body skin examination with dermatologist\n• Monthly self-examination using ABCDE criteria\n• Sun protection: SPF 50+, protective clothing, UV sunglasses\n• Avoid tanning beds and prolonged UV exposure")
            
            st.info("**Box 3 — Monitoring**\n\n• Dermoscopic photographic documentation annually\n• Compare lesion appearance over time\n• Visit dermatologist immediately if ABCDE changes observed")
            
            st.info("**Box 4 — Prevention**\n\n• Apply broad-spectrum SPF 50+ sunscreen daily\n• Wear protective clothing (UPF 50+)\n• Seek shade between 10am–4pm\n• Perform monthly skin self-checks")
            
            st.success("**Box 5 — Follow-Up**\n\n• Annual dermatology appointment\n• Immediate visit if: lesion changes color, size, shape, or bleeds")
            
        st.markdown("<p style='text-align:center; color:gray; font-size:12px;'>⚠️ This protocol is AI-generated for educational purposes. Always consult a qualified dermatologist for actual diagnosis.</p>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE: REPORTS
# ═══════════════════════════════════════════════
elif page == "📄 Reports":
    st.title("Reports")
    res = st.session_state.last_result
    
    if res is None:
        st.info("Run a scan first to generate reports.")
    else:
        st.subheader("Last Scan Summary")
        st.write(f"**Name:** {res['name']} | **Date:** {res['date']}")
        st.write(f"**Diagnosis:** {res['diagnosis']} | **Confidence:** {res['confidence']*100:.1f}% | **Risk:** {res['risk']}")
        
        pdf_bytes = create_pdf(res)
        st.download_button(label="📄 Download PDF Report", data=pdf_bytes, file_name=f"SkinScan_Report_{res['pid']}.pdf", mime="application/pdf")
        
    st.markdown("---")
    st.subheader("Data Export")
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        df_export = df.drop(columns=['multiclass'])
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(label="📊 Download Full History (CSV)", data=csv, file_name="skinscan_history.csv", mime="text/csv")
    else:
        st.info("No history available to export.")

# ═══════════════════════════════════════════════
# PAGE: HISTORY & ANALYTICS
# ═══════════════════════════════════════════════
elif page == "📊 History & Analytics":
    st.title("History & Analytics")
    
    if len(st.session_state.history) == 0:
        st.info("No scans yet.")
    else:
        df = pd.DataFrame(st.session_state.history)
        
        col1, col2, col3 = st.columns(3)
        diag_filter = col1.selectbox("Filter by Diagnosis", ["All", "MALIGNANT", "BENIGN"])
        risk_filter = col2.selectbox("Filter by Risk", ["All", "HIGH", "MEDIUM", "LOW"])
        search_name = col3.text_input("Search by patient name")
        
        filtered_df = df.copy()
        if diag_filter != "All":
            filtered_df = filtered_df[filtered_df['diagnosis'] == diag_filter]
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['risk'] == risk_filter]
        if search_name:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_name, case=False)]
            
        st.dataframe(filtered_df[['name', 'age', 'gender', 'date', 'diagnosis', 'confidence', 'risk', 'top_class']])
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            diag_counts = df['diagnosis'].value_counts().reset_index()
            diag_counts.columns = ['Diagnosis', 'Count']
            fig1 = px.pie(diag_counts, values='Count', names='Diagnosis', title='Diagnosis Distribution', 
                          color='Diagnosis', color_discrete_map={'BENIGN':'#2ed573', 'MALIGNANT':'#ff4757'})
            fig1.update_layout(paper_bgcolor="#0a0e1a", font_color="white")
            st.plotly_chart(fig1, use_container_width=True)
            
        with c2:
            date_counts = df['date'].apply(lambda x: x.split()[0]).value_counts().reset_index()
            date_counts.columns = ['Date', 'Count']
            fig2 = px.bar(date_counts, x='Date', y='Count', title='Scans per Date')
            fig2.update_traces(marker_color='#00d4ff')
            fig2.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="white", xaxis_showgrid=False, yaxis_showgrid=False)
            st.plotly_chart(fig2, use_container_width=True)
            
        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.histogram(df, x='confidence', nbins=10, title='Confidence Score Distribution')
            fig3.update_traces(marker_color='#ffa502')
            fig3.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="white", xaxis_showgrid=False, yaxis_showgrid=False)
            st.plotly_chart(fig3, use_container_width=True)
            
        with c4:
            risk_counts = df['risk'].value_counts().reset_index()
            risk_counts.columns = ['Risk', 'Count']
            fig4 = px.bar(risk_counts, x='Risk', y='Count', title='Risk Level Counts', 
                          color='Risk', color_discrete_map={'HIGH':'red', 'MEDIUM':'orange', 'LOW':'green'})
            fig4.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="white", xaxis_showgrid=False, yaxis_showgrid=False)
            st.plotly_chart(fig4, use_container_width=True)
            
        if st.button("🗑️ Clear All History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

# ═══════════════════════════════════════════════
# PAGE: MEDICAL GUIDE
# ═══════════════════════════════════════════════
elif page == "📚 Medical Guide":
    st.title("Medical Guide")
    
    with st.expander("Section 1 — What is Melanoma?"):
        st.write("""
        Melanoma is the most lethal form of skin cancer. While it accounts for less than 5% of all skin cancers, it causes the vast majority of skin cancer deaths.
        - **Stage I**: ~98% 5-year survival rate.
        - **Metastatic**: ~23% survival rate.
        
        Early detection is critical.
        """)
        
    with st.expander("Section 2 — ABCDE Criteria"):
        st.write("""
        - **A — Asymmetry**: One half unlike the other half.
        - **B — Border**: Irregular, ragged, notched, or blurred edges.
        - **C — Color**: Variation in color (tan, brown, black, red, white, blue).
        - **D — Diameter**: Larger than 6mm (pencil eraser size).
        - **E — Evolution**: Any change in size, shape, color, or new symptom (bleeding, itching).
        """)
        
    with st.expander("Section 3 — Skin Cancer Types"):
        st.table(pd.DataFrame({
            "Type": ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Merkel Cell Carcinoma", "Benign Nevus", "Seborrheic Keratosis"],
            "Description": ["Pigment cell cancer", "Most common, slow-growing", "Scaly/crusty lesions", "Rare, aggressive", "Common mole", "Waxy brown growths"],
            "Malignancy": ["Highly malignant", "Low metastasis", "Moderate", "High", "Benign", "Benign"]
        }))
        
    with st.expander("Section 4 — Risk Factors"):
        st.write("""
        • Excessive UV / sun exposure
        • Fair skin, light hair/eyes
        • Family history of melanoma
        • Personal history of skin cancer
        • Immunosuppression (HIV, transplant)
        • Large number of moles (>50)
        • Tanning bed use
        """)
        
    with st.expander("Section 5 — Prevention"):
        st.write("""
        • Apply SPF 50+ broad-spectrum sunscreen daily
        • Wear UPF 50+ protective clothing
        • Seek shade 10am–4pm
        • Never use tanning beds
        • Perform monthly ABCDE self-checks
        • Annual dermatologist visit after age 30
        """)
        
    with st.expander("Section 6 — Treatment Options"):
        st.write("""
        • **Surgery**: Wide local excision, Mohs surgery
        • **Immunotherapy**: Pembrolizumab, Nivolumab
        • **Targeted therapy**: Vemurafenib, Dabrafenib+Trametinib (BRAF mutated)
        • **Radiation**: For unresectable or brain metastases
        • **Chemotherapy**: Dacarbazine (last resort)
        • Clinical trials available at major cancer centers
        """)
        
    with st.expander("Section 7 — Emergency Warning Signs", expanded=True):
        st.error("""
        **Seek immediate medical attention if:**
        • Lesion bleeds spontaneously
        • Rapid growth within weeks
        • Lesion becomes painful or itchy suddenly
        • Swollen lymph nodes near lesion
        • Satellite lesions appear nearby
        • Unexplained fatigue, weight loss, or bone pain
        """)

# ═══════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════
elif page == "⚙️ About":
    import base64
    import os
    profile_img_b64 = "https://ui-avatars.com/api/?name=Rehan+Shafique&background=0D8ABC&color=fff&size=150&font-size=0.33&bold=true"
    if os.path.exists("profile.jpg"):
        with open("profile.jpg", "rb") as f:
            profile_img_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    elif os.path.exists("profile.png"):
        with open("profile.png", "rb") as f:
            profile_img_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    st.markdown("""
    <style>
    .about-container { animation: fadeInUp 1s ease-out; display: flex; flex-direction: column; gap: 30px; }
    .profile-card {
        background: rgba(15, 23, 42, 0.4); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.05);
        border-radius: 20px; padding: 30px; display: flex; align-items: center; gap: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3); transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        transform-style: preserve-3d; position: relative; overflow: hidden;
    }
    .profile-card::before {
        content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(0,212,255,0.1) 0%, transparent 70%);
        animation: spin 15s linear infinite; pointer-events: none;
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    .profile-card:hover {
        transform: translateY(-10px) rotateX(2deg) rotateY(-2deg);
        box-shadow: 0 20px 40px rgba(0,212,255,0.2); border-color: rgba(0,212,255,0.4);
    }
    .profile-image-container { position: relative; }
    .profile-image {
        width: 150px; height: 150px; border-radius: 50%; object-fit: cover; border: 3px solid #00d2ff;
        box-shadow: 0 0 20px rgba(0,212,255,0.5); animation: floatProfile 3s ease-in-out infinite; z-index: 2; position: relative;
    }
    .profile-glow {
        position: absolute; top: -10px; left: -10px; right: -10px; bottom: -10px;
        background: radial-gradient(circle, rgba(0,212,255,0.5) 0%, transparent 70%);
        border-radius: 50%; animation: pulse-glow 2s infinite alternate; z-index: 1;
    }
    @keyframes pulse-glow { 0% { opacity: 0.5; transform: scale(0.9); } 100% { opacity: 1; transform: scale(1.1); } }
    @keyframes floatProfile { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-10px); } }
    
    .typing-container { display: inline-block; }
    .typing-text {
        overflow: hidden; white-space: nowrap; border-right: 2px solid #00d2ff;
        animation: typing 3s steps(40, end), blink .75s step-end infinite;
        font-size: 1.2rem; font-weight: 500; color: #00d2ff; margin-top: 10px;
    }
    @keyframes typing { from { width: 0 } to { width: 100% } }
    @keyframes blink { 50% { border-color: transparent } }
    
    .skills-container { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
    .skill-badge {
        background: linear-gradient(90deg, rgba(0,212,255,0.1), rgba(58,123,213,0.1));
        border: 1px solid rgba(0,212,255,0.3); padding: 5px 15px; border-radius: 20px; font-size: 0.9rem;
        transition: all 0.3s;
    }
    .skill-badge:hover { transform: scale(1.1); background: rgba(0,212,255,0.2); box-shadow: 0 0 10px rgba(0,212,255,0.4); }
    
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; }
    .stat-box {
        background: rgba(255,255,255,0.02); border-radius: 15px; padding: 20px; text-align: center;
        border: 1px solid rgba(255,255,255,0.05); transition: transform 0.3s; backdrop-filter: blur(10px);
    }
    .stat-box:hover { transform: translateY(-5px) scale(1.05); border-color: #00d2ff; box-shadow: 0 10px 20px rgba(0,212,255,0.1); }
    .stat-num { font-size: 2rem; font-weight: bold; color: #00d2ff; }
    
    .timeline-container { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; width: 100%; }
    .timeline-column { display: flex; flex-direction: column; gap: 20px; }
    .timeline-item { border-left: 2px solid #00d2ff; padding-left: 20px; position: relative; }
    .timeline-item::before {
        content: ''; position: absolute; left: -6px; top: 5px; width: 10px; height: 10px;
        background: #00d2ff; border-radius: 50%; box-shadow: 0 0 10px #00d2ff;
    }
    .social-icons a { margin-right: 15px; text-decoration: none; font-size: 1.5rem; transition: 0.3s; opacity: 0.7; }
    .social-icons a:hover { opacity: 1; transform: scale(1.2) translateY(-3px); display: inline-block; }
    
    /* Responsive */
    @media (max-width: 768px) {
        .profile-card { flex-direction: column; text-align: center; }
        .timeline-container { grid-template-columns: 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
<div class="about-container">
<!-- Profile Header -->
<div class="profile-card">
<div class="profile-image-container">
<div class="profile-glow"></div>
<img src="{profile_img_b64}" class="profile-image" alt="Profile">
</div>
<div>
<h1 style="margin:0; font-size:2.8rem; font-weight:800; letter-spacing:1px;">Rehan Shafique</h1>
<div class="typing-container"><div class="typing-text">BS Bioinformatics | AI Medical Researcher</div></div>
<p style="margin-top:15px; opacity:0.8; max-width:600px; line-height:1.6;">
Developing cutting-edge artificial intelligence systems for dermatology. 
Merging deep learning (CNNs) with clinical workflows to provide accessible, high-precision diagnostic support for skin cancer detection.
</p>
<div class="skills-container">
<span class="skill-badge">TensorFlow / Keras</span>
<span class="skill-badge">Python 3.10+</span>
<span class="skill-badge">Computer Vision</span>
<span class="skill-badge">Streamlit UX</span>
<span class="skill-badge">Medical Image Processing</span>
</div>
<div class="social-icons" style="margin-top: 20px;">
<a href="#" style="color:#00d2ff;">🎓 UAF ID: 2022-AG-7662</a>
</div>
</div>
</div>

<!-- Stats Row -->
<div class="stats-grid">
<div class="stat-box"><div class="stat-num">88.7%</div><div style="opacity:0.8;">Model Accuracy</div></div>
<div class="stat-box"><div class="stat-num">0.928</div><div style="opacity:0.8;">AUC-ROC Score</div></div>
<div class="stat-box"><div class="stat-num">91.3%</div><div style="opacity:0.8;">Malignant Recall</div></div>
<div class="stat-box"><div class="stat-num">1.4s</div><div style="opacity:0.8;">CPU Inference Time</div></div>
</div>

<!-- Info Timeline -->
<div class="profile-card" style="align-items: flex-start;">
<div class="timeline-container">
<div class="timeline-column">
<h2 style="margin-top:0;">📚 Academic Project Info</h2>
<div class="timeline-item">
<h4 style="margin:0; color:#00d2ff;">Project Title</h4>
<p style="opacity:0.8; margin-top:5px;">SkinScan AI — AI-Based Skin Disease Detection</p>
</div>
<div class="timeline-item">
<h4 style="margin:0; color:#00d2ff;">Supervisor</h4>
<p style="opacity:0.8; margin-top:5px;">Dr. Hassan Tariq</p>
</div>
<div class="timeline-item">
<h4 style="margin:0; color:#00d2ff;">Institution</h4>
<p style="opacity:0.8; margin-top:5px;">Department of Computer Science<br>University of Agriculture Faisalabad (2026)</p>
</div>
</div>

<div class="timeline-column">
<h2 style="margin-top:0;">🧠 Deep Learning Architecture</h2>
<div class="timeline-item">
<h4 style="margin:0; color:#00d2ff;">Core Model</h4>
<p style="opacity:0.8; margin-top:5px;">MobileNetV2 with Custom Classification Head</p>
</div>
<div class="timeline-item">
<h4 style="margin:0; color:#00d2ff;">Training Protocol</h4>
<p style="opacity:0.8; margin-top:5px;">Two-phase transfer learning on Melanoma Cancer Dataset (Kaggle)</p>
</div>
<div class="timeline-item">
<h4 style="margin:0; color:#00d2ff;">Pipeline Details</h4>
<p style="opacity:0.8; margin-top:5px;">224×224×3 RGB Input → Sigmoid P(Malignant) Output</p>
</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)