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
    "Tomato___healthy":                               "Healthy",
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
        "description": "Bacterial Spot is caused by the bacterium Xanthomonas campestris. It appears as small, dark, water-soaked spots on leaves, stems, and fruits. The spots may have a yellow halo around them. It spreads rapidly in warm, wet conditions and can cause significant yield loss if left untreated.",
        "severity": "Moderate",
        "advice": "Remove and destroy infected leaves immediately. Apply copper-based bactericide as a preventive spray. Avoid overhead watering — water at the base of the plant. Rotate crops each season to prevent soil contamination.",
    },
    "Tomato___Early_blight": {
        "description": "Early Blight is caused by the fungus Alternaria solani. It is one of the most common tomato diseases, characterized by dark brown spots with concentric rings that form a target-like pattern. It usually starts on older, lower leaves and works its way up the plant as the season progresses.",
        "severity": "Moderate",
        "advice": "Remove infected lower leaves to slow the spread. Apply fungicide (chlorothalonil or copper-based) every 7–10 days. Ensure good air circulation by pruning. Mulch around plants to prevent soil splash.",
    },
    "Tomato___healthy": {
        "description": "The leaf shows no signs of disease or pest damage. The plant appears to be in good health with normal coloration and leaf structure. Continue with your current care routine to maintain plant health throughout the growing season.",
        "severity": "None",
        "advice": "Continue regular watering and fertilization. Monitor weekly for early signs of disease. Ensure adequate spacing between plants for airflow. Consider preventive copper sprays if disease pressure is high in your area.",
    },
    "Tomato___Late_blight": {
        "description": "Late Blight is caused by Phytophthora infestans, the same pathogen responsible for the Irish Potato Famine. It is one of the most destructive tomato diseases. It appears as large, irregular, water-soaked lesions that turn brown with a white mold on the underside of leaves. It can destroy an entire crop within days under favorable conditions.",
        "severity": "Severe",
        "advice": "Act immediately — Late Blight spreads extremely fast in cool, wet weather. Remove and bag all infected plant material (do not compost). Apply fungicide containing mancozeb or chlorothalonil right away. In severe cases, consider removing the entire plant to protect others nearby.",
    },
    "Tomato___Leaf_Mold": {
        "description": "Leaf Mold is caused by the fungus Passalora fulva (formerly Cladosporium fulvum). It appears as pale yellow or greenish spots on the upper leaf surface, with an olive-green to brown velvety mold growth on the underside. It thrives in high humidity environments, especially in greenhouses.",
        "severity": "Moderate",
        "advice": "Improve air circulation by pruning and spacing plants properly. Reduce humidity — avoid wetting foliage. Apply fungicide if the outbreak is severe. Remove and destroy heavily infected leaves.",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Septoria Leaf Spot is caused by the fungus Septoria lycopersici. It appears as numerous small, circular spots with dark brown borders and lighter tan or gray centers, often with tiny black dots (pycnidia) visible in the center. It starts on lower leaves and moves upward, causing significant defoliation.",
        "severity": "Moderate",
        "advice": "Remove infected leaves at first sign of disease. Apply fungicide (chlorothalonil or copper-based). Avoid working with plants when wet to prevent spreading spores. Mulch soil to reduce water splash from the ground.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Two-Spotted Spider Mites (Tetranychus urticae) are tiny arachnids, not insects. They feed on leaf cells, causing a stippled, bronzed, or silvery appearance on leaves. In severe infestations, fine webbing may be visible on the undersides of leaves. They thrive in hot, dry conditions and can reproduce rapidly.",
        "severity": "Moderate",
        "advice": "Spray plants forcefully with water to dislodge mites from leaf undersides. Apply neem oil or insecticidal soap — focus on the undersides of leaves. Introduce natural predators like predatory mites if available. Avoid over-fertilizing with nitrogen, which encourages mite population growth.",
    },
    "Tomato___Target_Spot": {
        "description": "Target Spot is caused by the fungus Corynespora cassiicola. It produces brown lesions with concentric rings resembling a target or bullseye pattern, similar to Early Blight. It affects leaves, stems, and fruits. Lesions can merge and cause significant defoliation in severe cases.",
        "severity": "Moderate",
        "advice": "Apply fungicide (azoxystrobin or chlorothalonil) at first sign of disease. Remove heavily infected leaves. Improve plant spacing for better airflow. Avoid excessive nitrogen fertilization which promotes lush growth susceptible to infection.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Tomato Mosaic Virus (ToMV) is a highly contagious viral disease that causes a mosaic-like pattern of light and dark green (or yellow and green) on leaves. Infected leaves may also be distorted, curled, or reduced in size. There is no cure — the virus persists in infected plant material and soil for years.",
        "severity": "Severe",
        "advice": "There is no cure for viral infections. Remove and destroy all infected plants immediately — do not compost. Disinfect tools with bleach solution after handling infected plants. Control aphids which spread the virus. Wash hands thoroughly before handling healthy plants.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is a devastating viral disease transmitted by the silverleaf whitefly (Bemisia tabaci). Infected plants show upward curling and yellowing of leaves, stunted growth, and greatly reduced fruit production. Young plants infected early may produce no fruit at all.",
        "severity": "Severe",
        "advice": "There is no cure. Remove and destroy infected plants to prevent spread. Control whitefly populations using yellow sticky traps, insecticidal soap, or neem oil. Use reflective mulches to repel whiteflies. Plant resistant tomato varieties in future seasons.",
    },
}

SEVERITY_COLOR = {
    "None":     "green",
    "Moderate": "orange",
    "Severe":   "red",
}

LOW_CONFIDENCE_CLASSES = {"Tomato___Early_blight", "Tomato___Target_Spot"}
LOW_CONFIDENCE_THRESHOLD = 75
UNCERTAIN_THRESHOLD = 50


@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def is_likely_green(image: Image.Image) -> bool:
    """Rough check if image contains enough green to possibly be a leaf."""
    img_small = image.convert("RGB").resize((64, 64))
    arr = np.array(img_small).astype(float)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_pixels = np.sum((g > r) & (g > b) & (g > 60))
    total_pixels = 64 * 64
    return (green_pixels / total_pixels) > 0.15


def main():
    st.set_page_config(
        page_title="Tomato Leaf Disease Scanner",
        page_icon="🍅",
        layout="centered",
    )

    # Header
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
            <h1 style='font-size: 2rem; margin-bottom: 0;'>🍅 Tomato Leaf Disease Scanner</h1>
            <p style='color: gray; font-size: 1rem; margin-top: 0.3rem;'>
                Upload a photo of a tomato leaf to detect diseases and get treatment advice.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Info banner
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Take a clear, close-up photo** of a single tomato leaf.
        2. **Upload the image** using the button below (JPG or PNG).
        3. **View the result** — the model will identify the disease, show confidence, and recommend action.
        
        **Tips for best accuracy:**
        - Use natural lighting when taking the photo
        - Make sure the leaf fills most of the frame
        - Avoid blurry or dark images
        
        **Note:** This model is trained specifically on tomato leaves. Results on other plants or non-leaf images will not be meaningful.
        """)

    # Upload
    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear, well-lit photo of a single tomato leaf."
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded image", use_container_width=True)

        # Green check warning
        if not is_likely_green(image):
            st.warning(
                "⚠️ The uploaded image does not appear to contain a green leaf. "
                "This model is designed for tomato leaf images only. "
                "Results on other images (people, objects, non-plant photos) will not be accurate."
            )

        model = load_model()

        with st.spinner("🔍 Analyzing image..."):
            processed = preprocess_image(image)
            predictions = model.predict(processed, verbose=0)[0]
            top_idx = int(np.argmax(predictions))
            confidence = float(predictions[top_idx]) * 100
            predicted_class = CLASS_NAMES[top_idx]

        st.divider()

        # Low overall confidence warning
        if confidence < UNCERTAIN_THRESHOLD:
            st.warning(
                f"⚠️ The model is not confident in this prediction ({confidence:.1f}%). "
                "This may be because the image is not a tomato leaf, the photo quality is low, "
                "or the disease pattern is unusual. Please consult an agricultural expert."
            )

        # Result header
        display_name = DISPLAY_NAMES[predicted_class]
        info = DISEASE_INFO[predicted_class]
        severity = info["severity"]
        severity_color = SEVERITY_COLOR[severity]

        if predicted_class == "Tomato___healthy":
            st.success(f"✅ Result: **{display_name}**")
        elif severity == "Severe":
            st.error(f"🔴 Result: **{display_name}**")
        else:
            st.warning(f"🟠 Result: **{display_name}**")

        # Severity badge + confidence
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{confidence:.1f}%")
            st.progress(confidence / 100)
        with col2:
            st.metric("Severity Level", severity)
            st.markdown(
                f"<span style='color:{severity_color}; font-size: 1.5rem;'>{'●' * (1 if severity == 'None' else 2 if severity == 'Moderate' else 3)}</span>",
                unsafe_allow_html=True
            )

        # Low confidence for similar-looking classes
        if predicted_class in LOW_CONFIDENCE_CLASSES and confidence < LOW_CONFIDENCE_THRESHOLD:
            st.info(
                "ℹ️ Note: Early Blight and Target Spot have very similar visual patterns "
                "and are sometimes confused by the model. Consider consulting an expert if unsure."
            )

        st.divider()

        # Disease description
        st.subheader("📋 About this disease")
        st.markdown(info["description"])

        # Recommended action
        st.subheader("💊 Recommended Action")
        st.info(info["advice"])

        st.divider()

        # Top 3 predictions
        st.subheader("📊 Top 3 Predictions")
        top3_idx = np.argsort(predictions)[::-1][:3]
        for i, idx in enumerate(top3_idx):
            name = DISPLAY_NAMES[CLASS_NAMES[idx]]
            score = float(predictions[idx]) * 100
            bar_color = "#2ecc71" if i == 0 else "#95a5a6"
            st.markdown(f"**{i+1}. {name}** — {score:.1f}%")
            st.progress(score / 100)

        st.divider()

        # Disclaimer
        st.caption(
            "⚠️ Disclaimer: This tool is intended as a decision-support aid only. "
            "Always consult a qualified agricultural expert or plant pathologist for "
            "confirmation and treatment decisions, especially for severe cases."
        )


if __name__ == "__main__":
    main()