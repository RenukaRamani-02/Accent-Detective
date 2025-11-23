# app.py — Accent Detective (HuBERT) — upload + browser-record + styled UI
import streamlit as st
import numpy as np
import librosa
import torch
import joblib
import tempfile, base64, os, requests
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from streamlit_lottie import st_lottie
from PIL import Image
import csv
import time
import soundfile as sf

# ---------------- page config ----------------
st.set_page_config(page_title="Accent Detective", page_icon="🎧", layout="wide")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
BG_PATH = os.path.join(ASSETS_DIR, "background.jpg")
PROJECT_PDF_PATH = "/mnt/data/Project_description.pdf"  # local project PDF (for your reference)

# ---------------- helpers ----------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_local(image_path, opacity=0.45):
    if not os.path.exists(image_path):
        return
    bin_str = get_base64_of_bin_file(image_path)
    css = f"""
    <style>
    .stApp {{
      background-image: url("data:image/jpg;base64,{bin_str}");
      background-size: cover;
      background-position: center;
      background-attachment: fixed;
    }}
    .bg-overlay {{
      position: fixed; top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(255,255,255,{opacity}); z-index: -1;
    }}
    .card {{ background: rgba(255,255,255,0.88); border-radius: 12px; padding: 18px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }}
    </style>
    <div class="bg-overlay"></div>
    """
    st.markdown(css, unsafe_allow_html=True)

def load_lottie_url(url):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# ---------------- header ----------------
set_background_local(BG_PATH, opacity=0.48)

col1, col2 = st.columns([1, 4])
with col1:
    if os.path.exists(LOGO_PATH):
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, width=220)
        except:
            pass
with col2:
    st.markdown("<h1 style='color:#0f1b4c;'>🎙️ Accent Detective</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#333;margin-top:-10px;'>Detect accent (HuBERT) & recommend cuisines</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- load models (cached) ----------------
@st.cache_resource
def load_models():
    # candidate filenames to try
    model_names = [
        "AccentDetective_HuBERT_model.pkl",
        "AccentDetective_model_balanced_rf.pkl",
        "AccentDetective_model.pkl",
    ]
    encoder_names = [
        "AccentDetective_HuBERT_label_encoder.pkl",
        "AccentDetective_HuBERT_encoder.pkl",
        "AccentDetective_label_encoder.pkl",
    ]
    clf = None
    le = None
    for m in model_names:
        if os.path.exists(m):
            # try joblib then pickle
            try:
                clf = joblib.load(m)
            except:
                import pickle
                with open(m, "rb") as f:
                    clf = pickle.load(f)
            break
    if clf is None:
        raise FileNotFoundError(f"No classifier .pkl found. Checked: {model_names}")

    for e in encoder_names:
        if os.path.exists(e):
            try:
                le = joblib.load(e)
            except:
                import pickle
                with open(e, "rb") as f:
                    le = pickle.load(f)
            break
    if le is None:
        raise FileNotFoundError(f"No label-encoder .pkl found. Checked: {encoder_names}")

    # load HuBERT feature extractor (transformers) and model
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()

    # cuisine mapping (keep and extend as needed)
    cuisine_map = {
        "andhra_pradesh": ["Pulihora", "Gongura Pachadi", "Pesarattu"],
        "tamil_nadu": ["Idli", "Sambar", "Pongal"],
        "karnataka": ["Bisi Bele Bath", "Ragi Mudde", "Mysore Pak"],
        "kerala": ["Appam", "Avial", "Puttu"],
        "telangana": ["Hyderabadi Biryani", "Kodi Kura", "Sarva Pindi"],
        # add more keys if your model has other labels (use lowercase underscores)
    }

    return clf, le, processor, hubert, cuisine_map

try:
    clf, le, processor, hubert, cuisine_map = load_models()
except Exception as e:
    st.error("Error loading model files: " + str(e))
    st.stop()

# ---------------- cuisine helpers (robust lookup) ----------------
def normalize_label_for_cuisine(label):
    if label is None:
        return None
    s = str(label).strip().lower()
    s = s.replace(" ", "_").replace("-", "_").replace(".", "").replace(",", "")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def find_cuisines_for_label(label, cuisine_map_local):
    key = normalize_label_for_cuisine(label)
    if not key:
        return ["Regional favorites"]
    # direct match
    if key in cuisine_map_local:
        return cuisine_map_local[key]
    # compact comparison
    compact = key.replace("_", "")
    for k in cuisine_map_local:
        if compact == k.replace("_", "") or compact in k.replace("_", "") or k.replace("_", "") in compact:
            return cuisine_map_local[k]
    # contains fallback
    for k in cuisine_map_local:
        if k in key or key in k:
            return cuisine_map_local[k]
    return ["Regional favorites"]

# ---------------- simple CSV logging ----------------
LOG_CSV = "predictions_log.csv"
def log_prediction(input_name, pred_label, confidence):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    header = ["timestamp", "input_file", "pred_label", "confidence"]
    exists = os.path.exists(LOG_CSV)
    try:
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(header)
            writer.writerow([now, input_name, pred_label, confidence])
    except Exception:
        pass

# ---------------- sidebar Lottie ----------------
with st.sidebar:
    lottie_url = "https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json"
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        try:
            st_lottie(lottie_json, height=260, key="accent-lottie")
        except:
            pass
    st.write("How to use:")
    st.write("- Record (browser) or upload a short English sentence (2–8s).")
    st.write("- Quiet room, speak clearly for best results.")
    if os.path.exists(PROJECT_PDF_PATH):
        st.markdown(f"[Project brief]({PROJECT_PDF_PATH})")

# ---------------- main UI ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
left, right = st.columns([1, 1.2])

with left:
    st.subheader("Input")
    input_mode = st.radio("Choose input:", ("🎤 Record (browser)", "📁 Upload file"))
    temp_path = None

    if input_mode == "📁 Upload file":
        uploaded = st.file_uploader("Upload audio (.wav/.mp3/.m4a/.ogg)", type=["wav","mp3","m4a","ogg","flac"])
        if uploaded is not None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tf.write(uploaded.read())
            tf.flush(); tf.close()
            temp_path = tf.name
            st.audio(temp_path)

    else:
        # prefer streamlit-audiorecorder if installed (works locally)
        try:
            from streamlit_audiorecorder import audiorecorder
            audio_bytes = audiorecorder("Click to record", "Stop and Save")
            if audio_bytes:
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tf.write(audio_bytes)
                tf.flush(); tf.close()
                temp_path = tf.name
                st.audio(temp_path)
        except Exception:
            # fallback: try st.audio_input (if available in this streamlit version)
            try:
                audio_bytes = st.audio_input("Record using your browser (click to start)")
                if audio_bytes is not None:
                    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    tf.write(audio_bytes.read())
                    tf.flush(); tf.close()
                    temp_path = tf.name
                    st.audio(temp_path)
            except Exception:
                st.info("Browser recording not available here — please upload a file instead.")
                st.write("To enable local browser recording: in your project venv run `pip install streamlit-audiorecorder` and restart Streamlit, then allow mic permission in the browser.")

with right:
    st.subheader("Controls & Output")
    if st.button("🔍 Detect Accent"):
        if not temp_path:
            st.error("Please record or upload audio first.")
        else:
            with st.spinner("Analyzing..."):
                # load audio reliably
                try:
                    audio, sr = librosa.load(temp_path, sr=16000)
                except Exception:
                    # fallback using soundfile
                    audio, sr = sf.read(temp_path)
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1)
                    if sr != 16000:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        sr = 16000

                # extract HuBERT features using transformers processor + model
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = hubert(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().reshape(1, -1)

                # predict accent
                try:
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(emb)[0]
                        idx = int(np.argmax(probs))
                        pred_label = le.inverse_transform([idx])[0]
                        confidence = float(probs.max() * 100)
                    else:
                        pred_label = le.inverse_transform(clf.predict(emb))[0]
                        confidence = None
                except Exception as e:
                    st.error("Prediction error: model/encoder may not match embeddings.")
                    st.text(repr(e))
                    st.stop()

                # map label -> cuisines (robust)
                cuisines = find_cuisines_for_label(pred_label, cuisine_map)

            # Display results (NO AGE DETECTION)
            st.markdown(f"### 🎯 Detected Accent: **{pred_label}**")
            if confidence is not None:
                st.markdown(f"**Confidence:** {confidence:.2f}%")
            else:
                st.markdown("**Confidence:** N/A")

            st.markdown("**Recommended Cuisines:**")
            for d in cuisines:
                st.write("- " + d)

            # dropdown to explore dishes
            selected = st.selectbox("Explore a dish (optional):", ["No, thanks"] + cuisines)
            if selected and selected != "No, thanks":
                st.info(f"Try ordering: **{selected}**")

            # log prediction
            try:
                log_prediction(os.path.basename(temp_path), pred_label, f"{confidence:.2f}%" if confidence is not None else "N/A")
            except:
                pass

            # cleanup temp file
            try:
                os.remove(temp_path)
            except:
                pass

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#333; margin-top:18px'>Built with ❤️ — Accent Detective</div>", unsafe_allow_html=True)

