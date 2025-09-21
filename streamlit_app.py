import streamlit as st
import cv2
import pytesseract
import spacy
import re
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os
import csv
import tempfile
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ---------------- NLP Model ----------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import en_core_web_sm
    nlp = en_core_web_sm.load()

# ---------------- India-specific PII patterns ----------------
PII_PATTERNS = {
    "AADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "MOBILE_NUMBER": r"\b(?:\+91|0)?[6-9]\d{9}\b",
    "IFSC": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "GSTIN": r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b",
    "VOTER_ID": r"\b[A-Z]{3}\d{7}\b",
    "DRIVING_LICENSE": r"\b[A-Z]{2}\d{2}\s?\d{11}\b",
    "VEHICLE_REGISTRATION": r"\b[A-Z]{2}\d{2}[A-Z]{0,2}\d{4}\b",
    "PIN_CODE": r"\b\d{6}\b",
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}",
    "URL": r"https?://\S+",
    "BANK_ACCOUNT": r"\b\d{9,18}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
}

# ---------------- Roles & Users ----------------
ROLES = {
    "admin": ["NAME","DOB","AADHAAR","PAN","MOBILE_NUMBER","EMAIL","IFSC","GSTIN","VOTER_ID","DRIVING_LICENSE"],
    "auditor": ["PAN","AADHAAR","MOBILE_NUMBER"],
    "viewer": []
}

USERS = {
    "admin_user": {"password": "admin123", "role": "admin"},
    "auditor_user": {"password": "audit123", "role": "auditor"}
}

# ---------------- PII Detection ----------------
def detect_pii(text):
    pii_entities = []
    doc = nlp(text)
    label_map = {
        "PERSON": "NAME", "DATE": "DOB", "GPE": "LOCATION", "ORG": "ORG",
        "NORP": "NORP", "LOC": "LOCATION", "FAC": "FAC", "TIME": "TIME",
        "MONEY": "MONEY", "QUANTITY": "QUANTITY", "CARDINAL": "CARDINAL",
        "LAW": "LAW"
    }
    for ent in doc.ents:
        entity_type = label_map.get(ent.label_, None)
        if entity_type:
            pii_entities.append((ent.text, entity_type))
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        pii_entities.extend([(m, label) for m in matches])
    return pii_entities

# ---------------- Image Redaction ----------------
def preprocess_image(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def redact_image(image, method="black", selected_pii_types=None):
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_img = preprocess_image(cv_img)
    data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
    detected_pii = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        word = data['text'][i].strip()
        if not word: continue
        line_num = data['line_num'][i]
        line_words = [data['text'][j] for j in range(n_boxes) if data['line_num'][j]==line_num]
        line_text = " ".join(line_words)
        entities = detect_pii(line_text)
        entities_to_redact = [ent for ent in entities if ent[1] in selected_pii_types]
        if entities_to_redact:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            detected_pii.extend([(ent_text, ent_label, (x, y, w, h)) for ent_text, ent_label in entities_to_redact])
            if method == "black":
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,0,0), -1)
            elif method == "blur":
                roi = cv_img[y:y+h, x:x+w]
                roi = cv2.GaussianBlur(roi, (51,51), 0)
                cv_img[y:y+h, x:x+w] = roi
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), detected_pii

# ---------------- Audit Logging ----------------
def log_redaction(username, role, filename, detected_pii):
    with open("audit_log.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        for ent_text, ent_label, _ in detected_pii:
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username, role, filename, ent_label, ent_text])

# ---------------- Privacy Risk ----------------
PII_WEIGHTS = {"AADHAAR":10,"PAN":8,"MOBILE_NUMBER":5,"EMAIL":4,"DOB":6,"NAME":3,"VOTER_ID":7,"DRIVING_LICENSE":7,"BANK_ACCOUNT":9,"IFSC":6}
def calculate_privacy_risk(detected_pii):
    return sum(PII_WEIGHTS.get(ent_label,1) for _, ent_label, _ in detected_pii)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📄 PII Redactor", layout="wide")
st.title("📄 PrivGuard-XR: Real-Time Privacy Guardian")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_role = "viewer"

# Sidebar Login
with st.sidebar:
    st.header("Login 🔑")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and password == USERS[username]["password"]:
            st.session_state.authenticated = True
            st.session_state.user_role = USERS[username]["role"]
            st.success(f"Logged in as {st.session_state.user_role}")
        else:
            st.error("Invalid credentials")

# Tabs
tabs = st.tabs(["📝 Redactor","📊 Privacy Score","📂 Audit Log"])

# ---------------- Tab 1: Redactor ----------------
with tabs[0]:
    uploaded_file = st.file_uploader("Upload an Image or PDF", type=["png","jpg","jpeg","pdf"])
    if uploaded_file:
        pii_options = list(PII_PATTERNS.keys()) + ["NAME","DOB","ORG","LOCATION"]
        selected_pii = st.multiselect("Select PII types to redact:", pii_options, default=pii_options)
        redaction_method = st.radio("Redaction Method:", ["black","blur"])
        admin_redaction_toggle = True
        if st.session_state.user_role == "admin":
            admin_redaction_toggle = st.checkbox("🔄 Redaction ON/OFF", value=True)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if uploaded_file.name.lower().endswith(".pdf"):
            try:
                pages = convert_from_path(tmp_path, poppler_path=None)
                redacted_pages, detected_pii = [], []
                for page in pages:
                    r_img, d_pii = redact_image(page, method=redaction_method, selected_pii_types=selected_pii)
                    redacted_pages.append(Image.fromarray(r_img))
                    detected_pii.extend(d_pii)
                st.image(redacted_pages[0], caption="Redacted PDF (Page 1)", width=600)
            except Exception as e:
                st.error(f"PDF Processing failed: {e}")
        else:
            image = Image.open(tmp_path)
            redacted_img, detected_pii = redact_image(image, method=redaction_method, selected_pii_types=selected_pii)
            st.image(redacted_img, caption="Redacted Image", width=600)

        log_redaction(username if st.session_state.authenticated else "anonymous",
                      st.session_state.user_role, uploaded_file.name, detected_pii)

# ---------------- Tab 2: Privacy Score ----------------
with tabs[1]:
    st.header("📊 Privacy Risk Summary")
    try:
        df = pd.read_csv("audit_log.csv", names=["Timestamp","Username","Role","Filename","PII_Label","PII_Text"], encoding="utf-8")
        if not df.empty:
            df['Score'] = df['PII_Label'].map(PII_WEIGHTS).fillna(1)
            scores = df.groupby("Filename")['Score'].sum().reset_index()
            for _, row in scores.iterrows():
                score = row['Score']
                if score >= 25: level, color = "🔥 Highly Sensitive", "#ff4b4b"
                elif score >= 10: level, color = "⚠️ Medium Sensitive", "#ffb84b"
                else: level, color = "✅ Low Sensitive", "#4bbf73"
                st.markdown(f"<div style='border:2px solid {color}; width:220px; padding:8px; border-radius:8px; background:#f9f9f9;'><b>{row['Filename']}</b><br/>Score: {score}<br/><span style='color:{color}'>{level}</span></div>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.info("No audit log yet.")

# ---------------- Tab 3: Audit Log ----------------
with tabs[2]:
    st.header("📂 Audit Log Viewer")
    try:
        df = pd.read_csv("audit_log.csv", names=["Timestamp","Username","Role","Filename","PII_Label","PII_Text"], encoding="utf-8")
        st.dataframe(df)
    except FileNotFoundError:
        st.info("No audit log yet.")
