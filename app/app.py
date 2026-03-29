import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
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

ADVICE = {
    "Tomato___Bacterial_spot":       "Remove infected leaves. Apply copper-based bactericide. Avoid overhead watering.",
    "Tomato___Early_blight":         "Remove lower infected leaves. Apply fungicide. Ensure good air circulation.",
    "Tomato___healthy":              "Your plant looks healthy! Continue regular watering and monitoring.",
    "Tomato___Late_blight":          "Act fast — Late Blight spreads quickly. Remove infected parts and apply fungicide immediately.",
    "Tomato___Leaf_Mold":            "Improve ventilation. Reduce humidity. Apply fungicide if severe.",
    "Tomato___Septoria_leaf_spot":   "Remove infected leaves. Apply fungicide. Avoid wetting leaves when watering.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spray with water to dislodge mites. Apply neem oil or miticide.",
    "Tomato___Target_Spot":          "Remove infected leaves. Apply fungicide. Avoid overhead irrigation.",
    "Tomato___Tomato_mosaic_virus":  "No cure — remove and destroy infected plants. Disinfect tools. Control aphids.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "No cure — remove infected plants. Control whiteflies which spread the virus.",
}

# Classes where model confidence can be lower — show a warning
LOW_CONFIDENCE_CLASSES = {"Tomato___Early_blight", "Tomato___Target_Spot"}

@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

def main():
    st.set_page_config(
        page_title="Tomato Leaf Disease Scanner",
        page_icon="🍅",
        layout="centered",
    )

    st.title("Tomato Leaf Disease Scanner")
    st.write("Upload a photo of a tomato leaf and the model will identify any disease.")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Choose a leaf image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            predictions = model.predict(processed, verbose=0)[0]
            top_idx = int(np.argmax(predictions))
            confidence = float(predictions[top_idx]) * 100
            predicted_class = CLASS_NAMES[top_idx]

        st.divider()

        # Result
        display_name = DISPLAY_NAMES[predicted_class]

        if predicted_class == "Tomato___healthy":
            st.success(f"Result: {display_name}")
        else:
            st.error(f"Result: {display_name}")

        # Confidence bar
        st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(confidence / 100)

        # Low confidence warning
        if predicted_class in LOW_CONFIDENCE_CLASSES and confidence < 75:
            st.warning(
                "This disease can look similar to others. "
                "Consider consulting an expert if unsure."
            )

        # Advice
        st.subheader("Recommended action")
        st.info(ADVICE[predicted_class])

        # Top 3 predictions
        st.subheader("Top 3 predictions")
        top3_idx = np.argsort(predictions)[::-1][:3]
        for i, idx in enumerate(top3_idx):
            name = DISPLAY_NAMES[CLASS_NAMES[idx]]
            score = predictions[idx] * 100
            st.write(f"{i+1}. {name} — {score:.1f}%")

if __name__ == "__main__":
    main()