import streamlit as st
import cv2
import pytesseract
import spacy
import re
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import os
import csv
import tempfile
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ---------------- Configuration ----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\aksha\OneDrive\Desktop\doc_redaction\doc_redaction\tesseract.exe"
POPPLER_PATH = None
if os.name == "nt":
    POPPLER_PATH = r"C:\### Domain\poppler-25.07.0\Library\bin"

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
    for ent in doc.ents:
        label_map = {
            "PERSON": "NAME",
            "DATE": "DOB",
            "GPE": "LOCATION",
            "ORG": "ORG",
            "NORP": "NORP",
            "LOC": "LOCATION",
            "FAC": "FAC",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "QUANTITY": "QUANTITY",
            "CARDINAL": "CARDINAL",
            "LAW": "LAW"
        }
        entity_type = label_map.get(ent.label_, None)
        if entity_type:
            pii_entities.append((ent.text, entity_type))
    for label, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        pii_entities.extend([(m, label) for m in matches])
    return pii_entities

# ---------------- Face Pixelation ----------------
def pixelate_face(cv_img, x, y, w, h, blocks=10):
    face_roi = cv_img[y:y+h, x:x+w]
    h_roi, w_roi = face_roi.shape[:2]
    x_steps = np.linspace(0, w_roi, blocks+1, dtype=int)
    y_steps = np.linspace(0, h_roi, blocks+1, dtype=int)
    for i in range(blocks):
        for j in range(blocks):
            x0, x1 = x_steps[i], x_steps[i+1]
            y0, y1 = y_steps[j], y_steps[j+1]
            roi = face_roi[y0:y1, x0:x1]
            color = roi.mean(axis=(0,1)).astype(np.uint8)
            face_roi[y0:y1, x0:x1] = color
    cv_img[y:y+h, x:x+w] = face_roi
    return cv_img

# ---------------- Image & PDF Redaction ----------------
def preprocess_image(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray

def redact_image(image, method="black", selected_pii_types=None, redact_faces=True):
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed_img = preprocess_image(cv_img)
    data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
    detected_pii_metadata = []

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        word = data['text'][i].strip()
        if not word:
            continue
        line_num = data['line_num'][i]
        line_words = [data['text'][j] for j in range(n_boxes) if data['line_num'][j]==line_num]
        line_text = " ".join(line_words)
        entities = detect_pii(line_text)
        entities_to_redact = [ent for ent in entities if ent[1] in selected_pii_types]
        if entities_to_redact:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            for ent_text, ent_label in entities_to_redact:
                detected_pii_metadata.append((ent_text, ent_label, (x, y, w, h)))
            if method == "black":
                cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,0,0), -1)
            elif method == "blur":
                roi = cv_img[y:y+h, x:x+w]
                roi = cv2.GaussianBlur(roi, (51,51), 0)  # heavier blur
                cv_img[y:y+h, x:x+w] = roi

    if redact_faces:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv_img = pixelate_face(cv_img, x, y, w, h, blocks=10)

    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), detected_pii_metadata

def overlay_pii(image, detected_pii):
    cv_img = np.array(image).copy()
    for ent_text, ent_label, (x,y,w,h) in detected_pii:
        cv2.rectangle(cv_img, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(cv_img, ent_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return Image.fromarray(cv_img)

def redact_pdf(input_pdf, method="black", selected_pii_types=None, redact_faces=True):
    pages = convert_from_path(input_pdf, poppler_path=POPPLER_PATH)
    redacted_pages = []
    all_detected_pii = []
    for page in pages:
        redacted_img, detected_pii = redact_image(page, method, selected_pii_types, redact_faces)
        redacted_pages.append(Image.fromarray(redacted_img))
        all_detected_pii.extend(detected_pii)
    return redacted_pages, all_detected_pii

def save_redacted_pdf(pages, output_path):
    image_list = [p.convert("RGB") for p in pages]
    image_list[0].save(output_path, save_all=True, append_images=image_list[1:])

# ---------------- Audit Logging ----------------
def log_redaction(username, role, filename, detected_pii):
    with open("audit_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        for ent_text, ent_label, _ in detected_pii:
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                username,
                role,
                filename,
                ent_label,
                ent_text
            ])

# ---------------- Privacy Risk ----------------
PII_WEIGHTS = {"AADHAAR":10,"PAN":8,"MOBILE_NUMBER":5,"EMAIL":4,"DOB":6,"NAME":3,"VOTER_ID":7,"DRIVING_LICENSE":7,"BANK_ACCOUNT":9,"IFSC":6}
def calculate_privacy_risk(detected_pii):
    score = 0
    for _, ent_label, _ in detected_pii:
        score += PII_WEIGHTS.get(ent_label,1)
    return score

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📄 PII Redactor", layout="wide")
st.title("📄 PrivGuard-XR: Real-Time Cross-Modal Privacy Guardian")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_role = "viewer"

# Sidebar Login
with st.sidebar:
    st.header("Login 🔑")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")
    if login_btn:
        if username in USERS and password == USERS[username]["password"]:
            st.session_state.authenticated = True
            st.session_state.user_role = USERS[username]["role"]
            st.success(f"Logged in as {st.session_state.user_role}")
        else:
            st.error("Invalid credentials")

# Tabs
tabs = st.tabs(["📝 Redactor","📊 Privacy Score","📂 Audit Log"])

# -------------------- Tab 1: Redactor --------------------
with tabs[0]:
    try:
        uploaded_file = st.file_uploader("Upload an Image or PDF", type=["png","jpg","jpeg","pdf"])
        if uploaded_file is None:
            st.info("Please upload a file to start redaction.")
        else:
            pii_options = list(PII_PATTERNS.keys()) + ["NAME","DOB","GENDER","ADDRESS","ORG","LOCATION","NORP","TIME","MONEY","QUANTITY","CARDINAL","LAW","FAC","IP_ADDRESS","URL"]
            selected_pii = st.multiselect("Select PII types to redact:", pii_options, default=pii_options)
            redaction_method = st.radio("Redaction Method for Text PII:", ["black","blur"])
            redact_faces = st.checkbox("Pixelate Faces", value=True)

            # Admin toggle
            admin_redaction_toggle = False
            if st.session_state.user_role == "admin":
                admin_redaction_toggle = st.checkbox("🔄 Redaction ON/OFF (Admin Preview)", value=True)

            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            user_role = st.session_state.user_role if st.session_state.authenticated else "viewer"
            allowed_labels = ROLES[user_role]

            # ---------------- REDACTION ----------------
            if uploaded_file.name.lower().endswith(".pdf"):
                try:
                    pages, detected_pii = redact_pdf(tmp_path, method=redaction_method,
                                                     selected_pii_types=selected_pii, redact_faces=redact_faces)
                    if admin_redaction_toggle is False and user_role=="admin":
                        pages = [overlay_pii(p, detected_pii) for p in pages]

                    output_file = f"redacted_{uploaded_file.name}"
                    save_redacted_pdf(pages, output_file)
                    st.success("✅ PDF Redacted!")

                    for i, page in enumerate(pages):
                        st.markdown(f"### Page {i+1}")
                        st.image(page, width=600)

                except Exception as e_pdf:
                    st.error(f"Failed to process PDF: {e_pdf}")
                    pages, detected_pii = [], []

            else:
                try:
                    image = Image.open(tmp_path)
                    redacted_img, detected_pii = redact_image(image, method=redaction_method,
                                                              selected_pii_types=selected_pii, redact_faces=redact_faces)
                    if admin_redaction_toggle is False and user_role=="admin":
                        redacted_img = overlay_pii(Image.fromarray(redacted_img), detected_pii)

                    final_img = Image.fromarray(redacted_img)
                    st.image(final_img, caption="Redacted Image with Role-based Overlay", width=600)
                    output_file = f"redacted_{uploaded_file.name}"
                    final_img.save(output_file)

                except Exception as e_img:
                    st.error(f"Failed to process image: {e_img}")
                    final_img = None
                    detected_pii = []

            # ---------------- Logging ----------------
            try:
                log_redaction(username if st.session_state.authenticated else "anonymous",
                              user_role, uploaded_file.name, detected_pii)
            except Exception as e_log:
                st.warning(f"Could not log audit data: {e_log}")

            # ---------------- Download Buttons ----------------
            try:
                if uploaded_file.name.lower().endswith(".pdf") and pages:
                    with open(output_file, "rb") as f:
                        st.download_button("📥 Download Redacted File", f, file_name=output_file, mime="application/pdf")
                elif final_img:
                    with open(output_file, "rb") as f:
                        st.download_button("📥 Download Redacted File", f, file_name=output_file, mime="image/png")
            except Exception as e_dl:
                st.warning(f"Could not create download button: {e_dl}")

            # Admin/or auditor original download
            if st.session_state.user_role in ["admin", "auditor"]:
                try:
                    with open(tmp_path, "rb") as f_orig:
                        st.download_button(
                            "🛡️ Download Original File (No Redactions)",
                            f_orig,
                            file_name=f"original_{uploaded_file.name}",
                            mime="application/pdf" if uploaded_file.name.lower().endswith(".pdf") else "image/png"
                        )
                except Exception as e_orig:
                    st.warning(f"Could not provide original file download: {e_orig}")

    except Exception as e_main:
        st.error(f"An unexpected error occurred: {e_main}")


# -------------------- Tab 2: Privacy Score --------------------
with tabs[1]:
    st.header("📊 Privacy Risk Summary")
    try:
        try:
            df = pd.read_csv("audit_log.csv", names=["Timestamp","Username","Role","Filename","PII_Label","PII_Text"], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv("audit_log.csv", names=["Timestamp","Username","Role","Filename","PII_Label","PII_Text"], encoding='latin1')

        if df.empty:
            st.info("Audit log is empty. No privacy score to display yet.")
        else:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Score'] = df['PII_Label'].map(PII_WEIGHTS).fillna(1)
            file_scores = df.groupby("Filename")['Score'].sum().reset_index().rename(columns={"Score":"Total_Score"})

            for idx, row in file_scores.iterrows():
                score = row['Total_Score']
                if score >= 25:
                    level = "🔥 Highly Sensitive"
                    color = "#ff4b4b"
                elif score >= 10:
                    level = "⚠️ Medium Sensitive"
                    color = "#ffb84b"
                else:
                    level = "✅ Low Sensitive"
                    color = "#4bbf73"

                st.markdown(f"""
                <div style='border:2px solid {color}; width:250px; padding:10px; border-radius:10px; background-color:#f9f9f9; margin-bottom:10px;'>
                    <h4 style='margin:0'>{row['Filename']}</h4>
                    <p style='margin:0'><b>Score:</b> {score}</p>
                    <p style='margin:0'><b>Level:</b> <span style='color:{color}'>{level}</span></p>
                </div>
                """, unsafe_allow_html=True)

                # Write PII details to txt file for download
                file_pii = df[df["Filename"] == row['Filename']]
                pii_txt_path = f"{row['Filename']}_pii.txt"
                with open(pii_txt_path, "w", encoding='utf-8') as f_txt:
                    for _, pii_row in file_pii.iterrows():
                        pii_score = PII_WEIGHTS.get(pii_row['PII_Label'], 1)
                        f_txt.write(f"{pii_row['PII_Label']} ({pii_score}) - {pii_row['PII_Text']}\n")
                with open(pii_txt_path, "rb") as f_txt:
                    st.download_button("📄 Download PII Details", f_txt, file_name=pii_txt_path)

    except FileNotFoundError:
        st.info("No audit log available yet.")
    except Exception as e:
        st.error(f"Error loading privacy score: {e}")

# -------------------- Tab 3: Audit Log --------------------
with tabs[2]:
    st.header("📂 Audit Log Viewer")
    try:
        try:
            df = pd.read_csv("audit_log.csv", names=["Timestamp","Username","Role","Filename","PII_Label","PII_Text"], encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv("audit_log.csv", names=["Timestamp","Username","Role","Filename","PII_Label","PII_Text"], encoding='latin1')

        if df.empty:
            st.info("Audit log is empty.")
        else:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            st.dataframe(df)

            st.markdown("### Filters")
            usernames = st.multiselect("Filter by User:", df["Username"].unique(), default=df["Username"].unique())
            pii_labels = st.multiselect("Filter by PII Label:", df["PII_Label"].unique(), default=df["PII_Label"].unique())
            filtered_df = df[(df["Username"].isin(usernames)) & (df["PII_Label"].isin(pii_labels))]
            st.dataframe(filtered_df)

            st.markdown("### Privacy Risk Stats")
            if not filtered_df.empty:
                risk_summary = filtered_df.groupby("Filename")["PII_Label"].count().reset_index().rename(columns={"PII_Label":"Count of PII"})
                st.bar_chart(risk_summary.set_index("Filename")["Count of PII"])
            else:
                st.info("No data for selected filters.")

    except FileNotFoundError:
        st.info("No audit log available yet.")
    except Exception as e:
        st.error(f"Error loading audit log: {e}")


