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

# ---------------- page config ----------------
st.set_page_config(page_title="Accent Detective", page_icon="🎧", layout="wide")

ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
BG_PATH = os.path.join(ASSETS_DIR, "background.jpg")

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
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=220)
with col2:
    st.markdown("<h1 style='color:#0f1b4c;'>🎙️ Accent Detective</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#333;margin-top:-10px;'>Detect accent (HuBERT), estimate age group & recommend cuisines</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- load models (cached) ----------------
@st.cache_resource
def load_models():
    # try several possible filenames (user may have named encoder/model differently)
    model_names = [
        "AccentDetective_HuBERT_model.pkl",
        "AccentDetective_HuBERT_model.pkl",
        "AccentDetective_model_balanced_rf.pkl",  # fallback
        "AccentDetective_model.pkl",
    ]
    encoder_names = [
        "AccentDetective_HuBERT_label_encoder.pkl",
        "AccentDetective_HuBERT_encoder.pkl",
        "AccentDetective_label_encoder_balanced.pkl",
        "AccentDetective_label_encoder.pkl",
        "AccentDetective_HuBERT_label_encoder.pkl"
    ]
    clf = None
    le = None
    # locate model
    for m in model_names:
        if os.path.exists(m):
            clf = joblib.load(m)
            break
    if clf is None:
        raise FileNotFoundError(f"No classifier .pkl found. Checked: {model_names}")

    for e in encoder_names:
        if os.path.exists(e):
            le = joblib.load(e)
            break
    if le is None:
        raise FileNotFoundError(f"No label-encoder .pkl found. Checked: {encoder_names}")

    # load feature-extractor + huBERT
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()
    # cuisine mapping
    cuisine_map = {
        "andhra_pradesh": ["Pulihora", "Gongura Pachadi", "Pesarattu"],
        "karnataka": ["Bisi Bele Bath", "Ragi Mudde", "Mysore Pak"],
        "kerala": ["Appam", "Avial", "Puttu"],
        "tamil": ["Idli", "Sambar", "Pongal"]
    }
    return clf, le, processor, hubert, cuisine_map

try:
    clf, le, processor, hubert, cuisine_map = load_models()
except Exception as e:
    st.error("Error loading model files: " + str(e))
    st.stop()

# ---------------- sidebar Lottie ----------------
with st.sidebar:
    lottie_url = "https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json"
    lottie_json = load_lottie_url(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=260, key="accent-lottie")
    st.write("How to use:")
    st.write("- Record (browser) or upload a short English sentence (2–8s).")
    st.write("- Quiet room, speak clearly for best results.")

# ---------------- main UI ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
left, right = st.columns([1, 1.2])

with left:
    st.subheader("Input")
    input_mode = st.radio("Choose input:", ("🎤 Record (browser)", "📁 Upload file"))
    temp_path = None

    if input_mode == "📁 Upload file":
        uploaded = st.file_uploader("Upload audio (.wav/.mp3/.m4a/.ogg)", type=["wav","mp3","m4a","ogg"])
        if uploaded is not None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tf.write(uploaded.read())
            tf.flush(); tf.close()
            temp_path = tf.name
            st.audio(temp_path)

    else:
        # try to use st.audio_input (available on recent Streamlit deployments)
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

with right:
    st.subheader("Controls & Output")
    if st.button("🔍 Detect Accent & Age"):
        if not temp_path:
            st.error("Please record or upload audio first.")
        else:
            with st.spinner("Analyzing..."):
                audio, sr = librosa.load(temp_path, sr=16000)
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = hubert(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().reshape(1, -1)

                # predict accent
                try:
                    probs = clf.predict_proba(emb)[0]
                    idx = int(np.argmax(probs))
                    pred_label = le.inverse_transform([idx])[0]
                    confidence = float(probs.max() * 100)
                except Exception:
                    # if classifier does not have predict_proba
                    pred_label = le.inverse_transform(clf.predict(emb))[0]
                    confidence = None

                # simple age heuristic (placeholder)
                energy = float(np.mean(np.abs(audio)))
                duration = len(audio) / sr
                age_score = int(18 + (np.clip(energy*30, 0, 1) + np.clip(duration/5, 0,1)) * 22)
                if age_score < 25:
                    age_group = "Young"
                elif age_score < 40:
                    age_group = "Adult"
                else:
                    age_group = "Senior"

                cuisines = cuisine_map.get(pred_label, ["Regional favorites"])

            st.markdown(f"### 🎯 Detected Accent: **{pred_label.title()}**")
            if confidence is not None:
                st.markdown(f"**Confidence:** {confidence:.2f}%")
            else:
                st.markdown("**Confidence:** N/A")
            st.markdown(f"**Predicted Age Group:** {age_group} ({age_score} yrs approx.)")
            st.markdown("**Recommended Cuisines:**")
            for d in cuisines:
                st.write("- " + d)

            # cleanup temp file
            try:
                os.remove(temp_path)
            except:
                pass

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#333; margin-top:18px'>Built with ❤️ — Accent Detective</div>", unsafe_allow_html=True)

