import streamlit as st
import numpy as np
import torch
import joblib
import os, csv, time, tempfile
import soundfile as sf
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import pandas as pd

st.set_page_config(page_title="Accent Detective", page_icon="🎧")

st.title("🎙️ Accent Detective — Indian Accent Identification (HuBERT)")
st.write("Upload a voice file to detect accent and age group. All predictions will be logged in predictions_log.csv")

LOG_CSV = "predictions_log.csv"

def log_prediction(file_name, accent, confidence, age_group):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    header = ["timestamp", "file", "accent", "confidence", "age_group"]
    exists = os.path.exists(LOG_CSV)

    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([now, file_name, accent, confidence, age_group])


@st.cache_resource
def load_hubert_and_model():
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert.eval()

    clf = joblib.load("AccentDetective_HuBERT_model.pkl")
    enc = joblib.load("AccentDetective_HuBERT_encoder.pkl")

    return processor, hubert, clf, enc


try:
    processor, hubert, clf, enc = load_hubert_and_model()
except Exception as e:
    st.error("Model loading error: " + str(e))
    st.stop()


def load_audio(path):
    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        return y
    except:
        y, _ = librosa.load(path, sr=16000)
        return y


def extract_hubert_features(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = hubert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


st.subheader("Upload Audio (WAV/MP3/M4A/OGG)")
uploaded = st.file_uploader("Upload file", type=["wav", "mp3", "m4a", "ogg"])

temp_path = None
if uploaded:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tf.write(uploaded.read())
    tf.close()
    temp_path = tf.name
    st.audio(temp_path)


if st.button("🔍 Detect Accent"):
    if not temp_path:
        st.error("Upload a file first!")
    else:
        with st.spinner("Analyzing..."):
            audio = load_audio(temp_path)
            emb = extract_hubert_features(audio)

            probs = clf.predict_proba(emb)[0]
            idx = int(np.argmax(probs))
            accent = enc.inverse_transform([idx])[0]
            confidence = round(float(probs[idx] * 100), 2)

            # Simple age heuristic
            energy = np.mean(np.abs(audio))
            if energy < 0.12:
                age_group = "Senior"
            elif energy < 0.25:
                age_group = "Adult"
            else:
                age_group = "Young"

        st.success(f"🎯 Accent: **{accent}**")
        st.info(f"Confidence: **{confidence}%**")
        st.write(f"Estimated Age Group: **{age_group}**")

        log_prediction(uploaded.name, accent, confidence, age_group)
        st.success("Logged into predictions_log.csv")


st.subheader("📄 Recent Logs")
if os.path.exists(LOG_CSV):
    df = pd.read_csv(LOG_CSV)
    st.dataframe(df.tail(10))

st.caption("Built with ❤️ — Accent Detective")

