import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from PIL import Image
import ollama
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# -------- Disease Model Loading --------
@st.cache_resource
def load_cnn_model():
    model = load_model('../Models/Disease/plant_disease_model.keras')
    with open('../Models/Disease/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    inv_class_indices = {v: k for k, v in class_indices.items()}
    return model, inv_class_indices

model, inv_class_indices = load_cnn_model()

# -------- Crop Model Loading --------
@st.cache_resource
def load_crop_model():
    model = pickle.load(open("../Models/All_Trained_Models/crop_model.pkl", "rb")) 
    with open("../Models/Crop_Recommandation/crop_mapping.json") as f:
        crop_mapping = json.load(f)
    crop_mapping = {int(k): v for k, v in crop_mapping.items()}
    return model, crop_mapping

crop_model, crop_mapping = load_crop_model()

# -------- Fertilizer Model Loading --------
@st.cache_resource
def load_fertilizer_model():
    model = pickle.load(open("../Models/All_Trained_Models/fertilizer_recommendation.pkl", "rb"))
    with open("../Models/Fertilizer_Recommandation/Model/fert_mapping.json") as f:
        fert_mapping = json.load(f)
    fert_mapping = {int(k): v for k, v in fert_mapping.items()}
    return model, fert_mapping

fert_model, fert_mapping = load_fertilizer_model()

# -------- Regional Dataset Loading --------
@st.cache_data
def load_regional_data():
    return pd.read_csv("../Models/Crop_Recommandation/Dataset/Regional_Data.csv") 

regional_df = load_regional_data()

# -------- Disease Info Function --------
def get_disease_info_from_ollama(disease_name: str, language="English", model_name='llama3'):
    language_prompts = {
        "English": "",
        "Hindi": "उत्तर हिंदी में दें।",
        "Marathi": "उत्तर मराठीत द्या."
    }

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

# -------- UI Layout --------
st.title("🌾 AI-Based Smart Farming")

tab1, tab2, tab3 = st.tabs(["Crop Disease Prediction", "Crop Recommendation", "Fertilizer Suggestion"])

# -------- Tab 1: Disease Prediction --------
with tab1:
    st.header("Crop Disease Prediction")
    language = st.selectbox("Select language for disease details", ["English", "Hindi", "Marathi"])
    uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", width=250)
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

            with st.spinner("🔍 Getting disease info from expert..."):
                try:
                    disease_info = get_disease_info_from_ollama(predicted_class_name, language=language)
                    st.subheader("📋 Disease Details")
                    st.markdown(disease_info)
                except Exception as e:
                    st.error("⚠️ Could not fetch disease info from Ollama.")
                    st.exception(e)

# -------- Tab 2: Crop Recommendation --------
with tab2:
    st.header("Crop Recommendation")

    st.subheader("📍 Regional Inputs")
    state = st.selectbox("Select State", regional_df["State_Name"].unique())
    district = st.selectbox("Select District", regional_df[regional_df["State_Name"] == state]["District_Name"].unique())
    season = st.selectbox("Select Season", regional_df["Season"].unique())

    st.subheader("🌱 Soil Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 90)
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        ph = st.number_input("pH", 0.0, 14.0, 6.5)
    with col2:
        P = st.number_input("Phosphorous (P)", 0, 200, 42)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    with col3:
        K = st.number_input("Potassium (K)", 0, 200, 43)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 200.0)

    if st.button("Recommend Crops"):
        soil_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        pred_probs = crop_model.predict_proba(soil_input)[0]
        top_indices = pred_probs.argsort()[-3:][::-1]
        ml_crops = [crop_mapping[i] for i in top_indices]

        filtered = regional_df[
            (regional_df["State_Name"] == state) &
            (regional_df["District_Name"] == district) &
            (regional_df["Season"] == season)
        ]
        regional_crops = sorted(filtered["Crop"].unique().tolist())

        st.success("✅ Recommended Crops Based on Soil (ML Prediction)")
        for i, crop in enumerate(ml_crops, 1):
            st.markdown(f"**{i}. {crop}**")

        st.success("📍 Regionally Suitable Crops")
        if regional_crops:
            for i, crop in enumerate(regional_crops, 1):
                st.markdown(f"**{i}. {crop}**")
        else:
            st.warning("No regional data found for the selected district and season.")

# -------- Tab 3: Fertilizer Suggestion --------
with tab3:
    st.header("Fertilizer Suggestion")

    st.subheader("🧪 Soil and Weather Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("Temperature (°C)", 0, 50, 30)
        nitrogen = st.number_input("Nitrogen", 0, 100, 25)
    with col2:
        humidity = st.number_input("Humidity (%)", 0, 100, 60)
        potassium = st.number_input("Potassium", 0, 100, 20)
    with col3:
        moisture = st.number_input("Moisture", 0, 100, 40)
        phosphorous = st.number_input("Phosphorous", 0, 100, 30)

    soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])

    if st.button("Recommend Fertilizer"):
        soil_map = {"Sandy": 0, "Loamy": 1, "Black": 2, "Red": 3, "Clayey": 4}
        soil_encoded = soil_map[soil_type]

        fert_input = np.array([[temp, humidity, moisture, soil_encoded, nitrogen, potassium, phosphorous]])
        fert_pred = fert_model.predict(fert_input)[0]
        fertilizer_name = fert_mapping.get(fert_pred, "Unknown")

        st.success(f"🧪 Recommended Fertilizer: **{fertilizer_name}**")
