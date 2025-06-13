import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sklearn
import numpy as np
import seaborn as sns
from ultralytics import YOLO # MODEL
import pytesseract #OCR MODEL
from PIL import Image
import tempfile
import re
from textblob import TextBlob #MODEL
import time 
from dotenv import load_dotenv
import os
import google.genai as genai
from deep_translator import GoogleTranslator # MODEL

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


st.set_page_config(page_title="MediMatch", page_icon=":pill:", layout="wide")

# Load the pre-trained YOLOv8 model
print(os.getcwd())
model = YOLO(os.path.join(os.getcwd(), 'best.pt'))


# -------- CSS STYLING ---------
with open("css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def ocr_on_pil_image(pil_img):
    # Convert PIL image to grayscale numpy array for OCR
    img_cv = np.array(pil_img.convert("RGB"))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    processed_img = Image.fromarray(gray)
    text = pytesseract.image_to_string(processed_img)
    return text.strip()

def find_medication_from_text(text, med_list):
    found_meds = []
    for med in med_list:
        if re.search(r'\b' + re.escape(med) + r'\b', text, re.IGNORECASE):
            found_meds.append(med)
    return found_meds if found_meds else ["Not found"]

# Medication database (for demonstration purposes)
medication_list = [
    "amlodipine", "paracetamol", "ibuprofen", "metformin", "atorvastatin",
    "lisinopril", "omeprazole", "simvastatin", "losartan", "amoxicillin", 
    "polymaltose", "levothyroxine", "hydrochlorothiazide", "gabapentin",
]

# load user history medication data
df = pd.read_csv("medications.csv")

if 'page' not in st.session_state:
    st.session_state.page = "home"

# --------------------------------- SIDEBAR SECTION -----------------------------------------------
st.sidebar.title("MENU")

st.sidebar.markdown("---")
if st.sidebar.button("  🏠  Home  "):
    st.session_state.page = "home"
if st.sidebar.button("💊 MediMatch"):
    st.session_state.page = "ai"
if st.sidebar.button(" 🧾 History "):
    st.session_state.page = "history"
if st.sidebar.button("🤖 chatbot"):
    st.session_state.page = "chatbot"


st.sidebar.markdown("---")

st.sidebar.subheader("About the Model")
st.sidebar.write("The AI uses a YOLO model (`best.pt`) for object detection.")

# Display the selected page
if st.session_state.page == "home":
    with st.container():
        st.markdown('<p class="MainHeader">MediMatch</p>', unsafe_allow_html=True)
        st.markdown('<p class="text">MediMatch is a smart web app that delivers key medication instructions to help users stay informed and take their medicines safely.</p>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4 = st.columns([5, 5, 5, 5])

        with col1:
            if st.button("Home"):
                st.session_state.page = "home"

        with col2:
            if st.button("MediMatch"):
                st.session_state.page = "ai"

        with col3:
            if st.button("History"):
                st.session_state.page = "history"
        with col4:
            if st.button("Chatbot"):
                st.session_state.page = "chatbot"

    st.markdown("""<hr style="height:2px;border:none;color:#4B8BBE;background-color:black;" />""", unsafe_allow_html=True)

    with st.container():
        st.markdown('<p class="MainHeader">Goal</p>', unsafe_allow_html=True)
        st.image("images/goals.png", use_container_width=True)
        st.markdown('<p class="text">The goal is to help elderly individuals who often struggle with identifying their medications, ensuring safer and easier medication management!</p>', unsafe_allow_html=True)


# ------------------------------------ AI SECTION -----------------------------------

elif st.session_state.page == "ai":
    st.markdown('<p class="MainHeader">MediMatch AI</p>', unsafe_allow_html=True)
    st.markdown("<p class='text'>MediMatch is an AI-powered object detection tool that allows users to upload images of their medication labels. It extracts key information and compares it with the user's past records to provide relevant insights.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an medication Label! (Keep the image aligned)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image via PIL, convert to OpenCV format
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)  # RGB format

        # Run YOLOv8 model
        results = model(image_np, conf=0.7)
        result = results[0]

        st.markdown("""<hr style="height:2px;border:none;color:#4B8BBE;background-color:black;" />""", unsafe_allow_html=True)

        st.markdown('<p class="subheader">Detected Label</p>', unsafe_allow_html=True)
        result_img = result.plot()
        st.image(result_img, use_container_width=True)

        # Extract detection data
        boxes = result.boxes
        print(boxes)
        print(boxes.xyxy)
        if boxes is not None and boxes.xyxy is not None and boxes.xyxy.shape[0] > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            cropped_images = []
            for i in range(len(cls)):
                confidence = conf[i]
                if confidence >= 0.7:
                    class_id = int(cls[i])
                    class_name = model.names[class_id]
                    box = xyxy[i].astype(int)

                    x1, y1, x2, y2 = box
                    cropped = image.crop((x1, y1, x2, y2))
                    cropped_images.append((class_name, cropped))

            st.markdown("""<hr style="height:2px;border:none;color:#4B8BBE;background-color:black;" />""", unsafe_allow_html=True)

           # Inside the loop where I process cropped images
            if len(cropped_images) > 0:
                st.markdown('<p class="subheader">Cropped Label</p>', unsafe_allow_html=True)
                for label, cropped_img in cropped_images:
                    st.image(cropped_img, use_container_width=False)

                    # Show spinner while processing OCR and extraction
                    with st.spinner('🔍 Extracting text and information...'):
                        text = ocr_on_pil_image(cropped_img)

                        # Spell Correction library
                        corrected_lines = []
                        for line in text.split('\n'):
                            blob = TextBlob(line)
                            corrected_lines.append(str(blob.correct()))
                        corrected_text = '\n'.join(corrected_lines).lower()

                        medication = find_medication_from_text(corrected_text, medication_list)

                        strength_match = re.search(r'(\d+\s*mg)', corrected_text, re.IGNORECASE)
                        strength = strength_match.group(1) if strength_match else "not found"

                        qty_match = re.search(r'\s+(tablet?s?|capsule?s?|pill?s?|tab?s?|cap?s?)', corrected_text, re.IGNORECASE)
                        quantity = qty_match.group(1) if qty_match else "not found"

                        storage_match = re.search(r'at\s+(\d+°\s*to\s*\d+°[CF])', corrected_text, re.IGNORECASE)
                        storage = storage_match.group(1) if storage_match else "not found"

                    # Display results after spinner completes
                    st.markdown("""<hr style="height:2px;border:none;color:#4B8BBE;background-color:black;" />""", unsafe_allow_html=True)
                    st.markdown('<p class="subheader">Extracted Information</p>', unsafe_allow_html=True)
                    st.markdown(f"**Medication:** {', '.join(medication)}")
                    st.markdown(f"**Strength:** {strength}")
                    st.markdown(f"**Quantity Type:** {quantity}")
                    st.markdown(f"**Storage Condition:** {storage}")
                    medication = medication[0]

                med_match = df['medication'].str.lower() == medication
                if med_match.any():
                    idx = df[med_match].index[0]

                    # For each column, update if new value is found
                    if medication.lower() != "not found":
                        df.at[idx, "medication"] = medication

                    if strength.lower() != "not found":
                        df.at[idx, "strength"] = strength

                    if quantity.lower() != "not found":
                        df.at[idx, "dosage form"] = quantity

                    if storage.lower() != "not found":
                        df.at[idx, "storage condition"] = storage

                else:
                    # Medication not found, append new row with extracted data (use "N/A" if Not found)
                    new_row = {
                        "medication": medication if medication.lower() != "not found" else "N/A",
                        "strength": strength if strength.lower() != "not found" else "N/A",
                        "dosage form": quantity if quantity.lower() != "not found" else "N/A",
                        "storage condition": storage if storage.lower() != "not found" else "N/A"
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                df.to_csv("medications.csv", index=False)
                st.markdown('<p style="font-size:15px; font-weight:bold; color:Black;">✅ Information updated successfully!</p>', unsafe_allow_html=True)
                st.markdown('<p class="text">To find out more about your medication, check out the chatbot!</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="text">❌ No objects detected</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="text">❌ No objects detected</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="text">Upload an image to start detection!</p>', unsafe_allow_html=True)

# ------------------------------------ HISTORY SECTION -----------------------------------

elif st.session_state.page == "history":
    st.markdown('<p class="MainHeader">Medication History</p>', unsafe_allow_html=True)
    st.markdown('<p class="text">Stay up to date with your medication history!</p>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)
    col1, col2 = st.columns([5, 5])

    with col1:
        st.markdown('<p class="subheader">Total Prescriptions</p>', unsafe_allow_html=True)
        image = Image.open("images/medications.png")
        st.image(image, use_container_width=True)
        st.markdown('<p class="text">The goal is to assist the elderly in identifying medications more easily, while also displaying the total count to support accurate tracking and usage.</p>', unsafe_allow_html=True)
            

    with col2:
        st.markdown('<p class="subheader">Upcoming refills</p>', unsafe_allow_html=True)
        image = Image.open("images/refills.png")
        st.image(image, use_container_width=True)
        st.markdown('<p class="text">This help the elderly identify medications, track total count, and stay informed about upcoming refills for better adherence.</p>', unsafe_allow_html=True)

elif st.session_state.page == "chatbot":
    st.markdown('<p class="MainHeader">Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="text">Chatbot is a smart AI-powered chatbot that can answer your questions about medications, side effects, and more.</p>', unsafe_allow_html=True)

    client = genai.Client(api_key=gemini_api_key)

    # Let user select a language
    language_map = {
        "English": "en",
        "Malay": "ms",
        "Chinese (Simplified)": "zh-CN",
        "Tamil (Indian)": "ta"
    }

    selected_med = st.selectbox("Select a medication from your history:", df['medication'].unique())

    selected_lang = st.selectbox("Select language for response:", list(language_map.keys()))
    lang_code = language_map[selected_lang]


    predefined_question = "Give a simplified overview of this medication for elderly users, covering its purpose, common side effects, and how to use it. Keep the explanation under 500 words. If the medication isn’t in the database, respond with: 'Medication not found in the database."

    if st.button("Generate Explanation"):
        selected_row = df[df['medication'] == selected_med].iloc[0]

        prompt = (
            f"Medication info: {selected_row.to_dict()}\n\n"
            f"Question: {predefined_question}\n"
            f"Answer:"
        )

        with st.spinner('🤖 Generating explanation...'):
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=500,
                    temperature=0.5,
                    top_p=0.95,
                    top_k=40,
                ),
                contents=prompt,
            )

        if lang_code != "en":
            try:
                translated_text = GoogleTranslator(source='en', target=lang_code).translate(response.text)
            except Exception as e:
                translated_text = f"Error translating: {e}"
        else:
            translated_text = response.text


        st.markdown("### Chatbot Response")
        st.write(translated_text)

            
# ---------------------------------------------- FOOTER SECTION --------------------------------------------------------
with st.container():
    st.markdown("""<hr style="height:2px;border:none;color:#4B8BBE;background-color:black;" />""", unsafe_allow_html=True)
    st.write("""
        Contact us: contact@MediMatch |    Follow us on [Twitter](https://twitter.com)    [Facebook](https://facebook.com)
        """)

