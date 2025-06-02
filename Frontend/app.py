import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from PIL import Image
import ollama

# Load model and class indices once at start
@st.cache_resource
def load_cnn_model():
    model = load_model('../Models/Disease/plant_disease_model.keras')
    with open('../Models/Disease/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    inv_class_indices = {v: k for k, v in class_indices.items()}
    return model, inv_class_indices

model, inv_class_indices = load_cnn_model()

st.title("🌾 AI-Based Smart Farming")

def get_disease_info_from_ollama(disease_name: str, language="English", model_name='llama3'):
    language_prompts = {
        "English": "",
        "Hindi": "उत्तर हिंदी में दें।",
        "Marathi": "उत्तर मराठीत द्या."
    }

    # Put language instruction at the start of the prompt
    prompt = (
        language_prompts.get(language, "") + "\n" +
        f"Give a detailed explanation of the plant disease '{disease_name}'. "
        "Include:\n"
        "1. The cause of the disease\n"
        "2. Remedies (organic and chemical)\n"
        "3. Prevention tips\n"
        "Answer clearly and concisely."
    )
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert plant disease consultant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['message']['content']

# Create tabs
tab1, tab2, tab3 = st.tabs(["Crop Disease Prediction", "Crop Recommendation", "Fertilizer Suggestion"])

with tab1:
    st.header("Crop Disease Prediction")

    # Language selector
    language = st.selectbox(
        "Select language for disease details",
        options=["English", "Hindi", "Marathi"]
    )

    uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", width=250)

        # Preprocess image
        img = image_pil.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Predict Disease"):
            prediction = model.predict(img_array)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = inv_class_indices[predicted_class_index]
            confidence = np.max(prediction)

            st.success(f"🧠 Predicted Disease: **{predicted_class_name}**")
            st.info(f"Confidence: {confidence:.2f}")

            # Get disease info from Ollama in selected language
            with st.spinner("🔍 Getting disease info from expert..."):
                try:
                    disease_info = get_disease_info_from_ollama(predicted_class_name, language=language)
                    st.subheader("📋 Disease Details")
                    st.markdown(disease_info)
                except Exception as e:
                    st.error("⚠️ Could not fetch disease info from Ollama.")
                    st.exception(e)

with tab2:
    st.header("Crop Recommendation")
    st.info("This module is under development. Crop recommendation model will be added soon.")

with tab3:
    st.header("Fertilizer Suggestion")
    st.info("This module is under development. Fertilizer suggestion model will be added soon.")
