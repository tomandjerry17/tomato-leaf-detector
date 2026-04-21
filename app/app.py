import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image

MODEL_PATH = "models/model.h5"

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

DISPLAY_NAMES = {
    "Tomato___Bacterial_spot":                        "Bacterial Spot",
    "Tomato___Early_blight":                          "Early Blight",
    "Tomato___healthy":                               "Healthy Leaf",
    "Tomato___Late_blight":                           "Late Blight",
    "Tomato___Leaf_Mold":                             "Leaf Mold",
    "Tomato___Septoria_leaf_spot":                    "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":  "Spider Mites",
    "Tomato___Target_Spot":                           "Target Spot",
    "Tomato___Tomato_mosaic_virus":                   "Tomato Mosaic Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":         "Yellow Leaf Curl Virus",
}

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "description": "Caused by the bacterium Xanthomonas campestris. Appears as small, dark, water-soaked spots on leaves, stems, and fruits — often surrounded by a yellow halo. Spreads rapidly in warm, wet conditions.",
        "severity": "Moderate",
        "severity_level": 2,
        "advice": [
            "Remove and destroy infected leaves immediately",
            "Apply copper-based bactericide as a preventive spray",
            "Avoid overhead watering — water at the base of the plant",
            "Rotate crops each season to prevent soil contamination",
        ],
    },
    "Tomato___Early_blight": {
        "description": "Caused by the fungus Alternaria solani. Characterized by dark brown spots with concentric rings forming a target-like pattern. Usually starts on older, lower leaves and works its way upward.",
        "severity": "Moderate",
        "severity_level": 2,
        "advice": [
            "Remove infected lower leaves to slow the spread",
            "Apply fungicide every 7–10 days",
            "Ensure good air circulation by pruning",
            "Mulch around plants to prevent soil splash",
        ],
    },
    "Tomato___healthy": {
        "description": "No signs of disease or pest damage detected. The plant appears to be in good health with normal coloration and leaf structure. Continue with your current care routine.",
        "severity": "Healthy",
        "severity_level": 0,
        "advice": [
            "Continue regular watering and fertilization",
            "Monitor weekly for early signs of disease",
            "Ensure adequate spacing between plants for airflow",
            "Consider preventive copper sprays if disease pressure is high",
        ],
    },
    "Tomato___Late_blight": {
        "description": "Caused by Phytophthora infestans — the same pathogen behind the Irish Potato Famine. One of the most destructive tomato diseases. Large, water-soaked lesions turn brown with white mold on leaf undersides. Can destroy an entire crop within days.",
        "severity": "Critical",
        "severity_level": 3,
        "advice": [
            "Act immediately — this spreads extremely fast in cool, wet weather",
            "Remove and bag all infected material (do not compost)",
            "Apply mancozeb or chlorothalonil fungicide right away",
            "Consider removing the entire plant to protect others nearby",
        ],
    },
    "Tomato___Leaf_Mold": {
        "description": "Caused by the fungus Passalora fulva. Appears as pale yellow spots on the upper leaf surface with olive-green velvety mold growth on the underside. Thrives in high humidity, especially in greenhouses.",
        "severity": "Moderate",
        "severity_level": 2,
        "advice": [
            "Improve air circulation by pruning and spacing plants",
            "Reduce humidity — avoid wetting foliage when watering",
            "Apply fungicide if the outbreak is severe",
            "Remove and destroy heavily infected leaves",
        ],
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Caused by the fungus Septoria lycopersici. Appears as numerous small, circular spots with dark brown borders and lighter tan or gray centers, often with tiny black dots visible inside. Starts on lower leaves and causes significant defoliation.",
        "severity": "Moderate",
        "severity_level": 2,
        "advice": [
            "Remove infected leaves at first sign of disease",
            "Apply chlorothalonil or copper-based fungicide",
            "Avoid working with plants when wet to prevent spreading spores",
            "Mulch soil to reduce water splash from the ground",
        ],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Two-Spotted Spider Mites (Tetranychus urticae) are tiny arachnids that feed on leaf cells, causing a stippled, bronzed, or silvery appearance. In severe infestations, fine webbing is visible on leaf undersides. Thrives in hot, dry conditions.",
        "severity": "Moderate",
        "severity_level": 2,
        "advice": [
            "Spray plants forcefully with water to dislodge mites",
            "Apply neem oil or insecticidal soap on leaf undersides",
            "Introduce natural predators like predatory mites if available",
            "Avoid over-fertilizing with nitrogen which encourages mite growth",
        ],
    },
    "Tomato___Target_Spot": {
        "description": "Caused by the fungus Corynespora cassiicola. Produces brown lesions with concentric rings resembling a target or bullseye pattern. Affects leaves, stems, and fruits. Lesions can merge and cause significant defoliation.",
        "severity": "Moderate",
        "severity_level": 2,
        "advice": [
            "Apply azoxystrobin or chlorothalonil fungicide at first sign",
            "Remove heavily infected leaves",
            "Improve plant spacing for better airflow",
            "Avoid excessive nitrogen fertilization",
        ],
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "A highly contagious viral disease causing a mosaic pattern of light and dark green or yellow patches on leaves. Infected leaves may be distorted or curled. There is no cure — the virus persists in plant material and soil for years.",
        "severity": "Critical",
        "severity_level": 3,
        "advice": [
            "No cure — remove and destroy all infected plants immediately",
            "Do not compost infected material",
            "Disinfect tools with bleach solution after handling infected plants",
            "Control aphids which spread the virus between plants",
        ],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "A devastating viral disease transmitted by the silverleaf whitefly. Causes upward curling and yellowing of leaves, stunted growth, and severely reduced fruit production. Young plants infected early may produce no fruit at all.",
        "severity": "Critical",
        "severity_level": 3,
        "advice": [
            "No cure — remove and destroy infected plants immediately",
            "Control whitefly populations with yellow sticky traps",
            "Apply neem oil or insecticidal soap to reduce whitefly spread",
            "Use reflective mulches to repel whiteflies",
        ],
    },
}

LOW_CONFIDENCE_CLASSES = {"Tomato___Early_blight", "Tomato___Target_Spot"}


@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def is_likely_leaf(image: Image.Image) -> bool:
    img_small = image.convert("RGB").resize((64, 64))
    arr = np.array(img_small).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_pixels = np.sum((g > r) & (g > b) & (g > 60))
    return (green_pixels / (64 * 64)) > 0.15


def apply_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background-color: #F4F6F1; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 760px; }

    /* Hero */
    .hero {
        background: #1A3528;
        border-radius: 20px;
        padding: 2.75rem 2.5rem 2.25rem;
        margin-bottom: 1.75rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: ''; position: absolute;
        top: -50px; right: -50px;
        width: 180px; height: 180px;
        border-radius: 50%;
        background: rgba(255,255,255,0.05);
    }
    .hero-tag {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        color: #9FCFB0;
        font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.14em; text-transform: uppercase;
        padding: 0.28rem 0.8rem; border-radius: 20px;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem; color: #FFFFFF;
        line-height: 1.2; margin: 0 0 0.6rem 0;
    }
    .hero-sub {
        color: #7DAE8E; font-size: 0.9rem;
        font-weight: 300; line-height: 1.65;
        margin: 0; max-width: 460px;
    }
    .hero-icon {
        position: absolute; top: 1.5rem; right: 2rem;
        font-size: 4.5rem; opacity: 0.12;
    }

    /* Cards */
    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8DF;
        border-radius: 16px;
        padding: 1.6rem 1.75rem;
        margin-bottom: 1.1rem;
    }
    .card-label {
        font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #7A9E85; margin-bottom: 0.6rem;
    }
    .disease-name {
        font-family: 'DM Serif Display', serif;
        font-size: 1.9rem; color: #1A3528;
        margin: 0.2rem 0 0.6rem; line-height: 1.2;
    }
    .disease-desc {
        color: #546158; font-size: 0.9rem;
        line-height: 1.72; margin: 0;
    }

    /* Badges */
    .badge {
        display: inline-block;
        font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        padding: 0.28rem 0.85rem; border-radius: 20px;
        margin-bottom: 0.35rem;
    }
    .badge-healthy  { background: #D6EEE0; color: #1A5C33; }
    .badge-moderate { background: #FEF0CC; color: #7A5A00; }
    .badge-critical { background: #FFE3DF; color: #A83228; }

    /* Confidence */
    .conf-wrap { margin: 1.1rem 0 0.4rem; }
    .conf-label {
        display: flex; justify-content: space-between;
        font-size: 0.8rem; color: #6A7D6E;
        font-weight: 500; margin-bottom: 0.35rem;
    }
    .conf-track {
        background: #EAF0E6; border-radius: 6px;
        height: 8px; overflow: hidden;
    }
    .conf-fill { height: 100%; border-radius: 6px; }
    .fill-high   { background: #2A7040; }
    .fill-medium { background: #C97E0A; }
    .fill-low    { background: #B83030; }

    /* Advice */
    .advice-list {
        border-left: 3px solid #2A7040;
        padding-left: 1.1rem;
        margin-top: 0.5rem;
    }
    .advice-list.critical { border-left-color: #B83030; }
    .advice-item {
        font-size: 0.9rem; color: #374039;
        line-height: 1.6; padding: 0.3rem 0;
        display: flex; gap: 0.55rem; align-items: flex-start;
    }
    .dot { color: #2A7040; font-size: 1rem; line-height: 1.5; flex-shrink: 0; }
    .dot.critical { color: #B83030; }

    /* Predictions */
    .pred-row {
        display: flex; align-items: center;
        gap: 0.85rem; margin-bottom: 0.75rem;
    }
    .pred-rank { font-size: 0.72rem; font-weight: 600; color: #A0B09A; width: 18px; flex-shrink: 0; }
    .pred-name { font-size: 0.86rem; color: #253028; font-weight: 500; flex: 1; }
    .pred-track { width: 110px; background: #EAF0E6; border-radius: 5px; height: 6px; overflow: hidden; flex-shrink: 0; }
    .pred-fill  { height: 100%; border-radius: 5px; background: #3A7A50; }
    .pred-fill.alt { background: #C0D4C4; }
    .pred-pct { font-size: 0.8rem; color: #6A7D6E; font-weight: 600; width: 42px; text-align: right; flex-shrink: 0; }

    /* Warnings */
    .warn {
        background: #FFFBEC; border: 1px solid #EDD060;
        border-radius: 12px; padding: 0.85rem 1.15rem;
        font-size: 0.86rem; color: #5C4900;
        line-height: 1.55; margin-bottom: 1.1rem;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.76rem; color: #8FA594;
        text-align: center; line-height: 1.65;
        border-top: 1px solid #DDE6DA;
        padding-top: 1.5rem; margin-top: 0.5rem;
    }

    /* Streamlit overrides */
    div[data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF !important;
        border: 2px dashed #C0D4BA !important;
        border-radius: 14px !important;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #2A7040 !important;
    }
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8DF !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def conf_bar_html(score: float, is_critical: bool = False) -> str:
    if score >= 70:
        fill = "fill-low" if is_critical else "fill-high"
    elif score >= 45:
        fill = "fill-medium"
    else:
        fill = "fill-low"
    return f"""
    <div class="conf-wrap">
        <div class="conf-label"><span>Model confidence</span><span>{score:.1f}%</span></div>
        <div class="conf-track">
            <div class="conf-fill {fill}" style="width:{min(score,100)}%"></div>
        </div>
    </div>"""


def main():
    st.set_page_config(
        page_title="TomatoScan — Leaf Disease Detector",
        page_icon="🍅",
        layout="centered",
    )
    apply_theme()

    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🍅</div>
        <div class="hero-tag">AI-Powered Plant Health</div>
        <h1 class="hero-title">Tomato Leaf<br>Disease Scanner</h1>
        <p class="hero-sub">Upload a photo of a tomato leaf to instantly detect diseases
        and receive expert-guided treatment recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How to use this tool"):
        st.markdown("""
        **For best results:**
        - Take a clear, close-up photo of a **single tomato leaf**
        - Use **natural lighting** and avoid heavy shadows or flash
        - Make sure the leaf **fills most of the frame**
        - Upload a **JPG or PNG** file

        **Note:** This model is trained specifically on tomato leaves.
        Uploading unrelated images will produce unreliable results.
        """)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload tomato leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <p style='text-align:center;color:#8FA594;font-size:0.85rem;padding:0.5rem 0'>
        Supported formats: JPG, JPEG, PNG</p>
        """, unsafe_allow_html=True)
        return

    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, use_container_width=True, caption="Uploaded image")

    if not is_likely_leaf(image):
        st.markdown("""
        <div class="warn">
        ⚠️ <strong>Image may not be a tomato leaf.</strong>
        This scanner is designed for tomato leaf images only.
        Results on unrelated photos will not be meaningful.
        </div>""", unsafe_allow_html=True)

    model = load_model()
    with st.spinner("Analyzing leaf..."):
        processed  = preprocess_image(image)
        predictions = model.predict(processed, verbose=0)[0]

    top_idx      = int(np.argmax(predictions))
    confidence   = float(predictions[top_idx]) * 100
    pred_class   = CLASS_NAMES[top_idx]
    info         = DISEASE_INFO[pred_class]
    display_name = DISPLAY_NAMES[pred_class]
    severity     = info["severity"]
    is_critical  = severity == "Critical"
    is_healthy   = severity == "Healthy"

    badge_cls = "badge-healthy" if is_healthy else "badge-critical" if is_critical else "badge-moderate"

    # Confidence warnings
    if confidence < 50:
        st.markdown(f"""
        <div class="warn">⚠️ <strong>Low confidence ({confidence:.1f}%).</strong>
        The model is uncertain about this prediction. The photo may be blurry, poorly lit,
        or not a tomato leaf. Please retake the photo or consult an agricultural expert.
        </div>""", unsafe_allow_html=True)
    elif pred_class in LOW_CONFIDENCE_CLASSES and confidence < 75:
        st.markdown("""
        <div class="warn">ℹ️ <strong>Similar disease patterns detected.</strong>
        Early Blight and Target Spot have visually similar concentric ring patterns
        and are sometimes confused by the model. Expert verification is recommended.
        </div>""", unsafe_allow_html=True)

    # Result card
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Diagnosis Result</div>
        <span class="badge {badge_cls}">{severity}</span>
        <h2 class="disease-name">{display_name}</h2>
        <p class="disease-desc">{info['description']}</p>
        {conf_bar_html(confidence, is_critical)}
    </div>""", unsafe_allow_html=True)

    # Advice card
    dot_cls  = "dot critical" if is_critical else "dot"
    list_cls = "advice-list critical" if is_critical else "advice-list"
    items_html = "".join(
        f'<div class="advice-item"><span class="{dot_cls}">›</span><span>{step}</span></div>'
        for step in info["advice"]
    )
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Recommended Action</div>
        <div class="{list_cls}">{items_html}</div>
    </div>""", unsafe_allow_html=True)

    # Top 3 predictions
    top3_idx = np.argsort(predictions)[::-1][:3]
    rows = ""
    for i, idx in enumerate(top3_idx):
        name  = DISPLAY_NAMES[CLASS_NAMES[idx]]
        score = float(predictions[idx]) * 100
        fill  = "pred-fill" if i == 0 else "pred-fill alt"
        rows += f"""
        <div class="pred-row">
            <span class="pred-rank">#{i+1}</span>
            <span class="pred-name">{name}</span>
            <div class="pred-track"><div class="{fill}" style="width:{score:.0f}%"></div></div>
            <span class="pred-pct">{score:.1f}%</span>
        </div>"""

    st.markdown(f"""
    <div class="card">
        <div class="card-label">Top Predictions</div>
        {rows}
    </div>""", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        This tool is a decision-support aid only and does not replace professional agricultural advice.<br>
        Always consult a qualified plant pathologist for confirmation and treatment decisions.<br><br>
        Model: MobileNetV2 + Transfer Learning &nbsp;·&nbsp; Accuracy: 86% &nbsp;·&nbsp; 10 disease classes
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()