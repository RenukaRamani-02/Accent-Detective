# Accent Detective — Final Streamlit Cloud Version (With CSV Logging)
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import pickle
import soundfile as sf
import tempfile, os, time, csv

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Accent Detective", page_icon="🎧", layout="wide")

st.title("🎙️ Accent Detective — Indian Accent & Age Prediction")
st.write("Upload or record audio. Every prediction is saved in **predictions_log.csv**.")

# ---------------- Logging ----------------
LOG_CSV = "predictions_log.csv"

def log_prediction(input_name, label, confidence, age_group):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    header = ["timestamp", "input_file", "predicted_label", "confidence", "age_group"]
    exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([now, input_name, label, confidence, age_group])

# ---------------- Load Model + Encoder Safely ----------------
@st.cache_resource
def load_models():
    clf_path = "AccentDetective_HuBERT_model.pkl"
    enc_path = "AccentDetective_HuBERT_encoder.pkl"

    # Load classifier
    try:
        clf = joblib.load(clf_path)
    except:
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)

    # Load label encoder
    try:
        enc = joblib.load(enc_path)
    except:
        with open(enc_path, "rb") as f:
            enc = pickle.load(f)

    return clf, enc

try:
    clf, enc = load_models()
except Exception as e:
    st.error("❌ Could not load model/encoder files.\n" + str(e))
    st.stop()

# ---------------- Audio Feature Extraction (MFCC) ----------------
def load_audio(path, sr=16000):
    try:
        y, sr_read = sf.read(path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr_read != sr:
            y = librosa.resample(y, orig_sr=sr_read, target_sr=sr)
        return y
    except:
        y, _ = librosa.load(path, sr=sr)
        return y

def extract_features(path):
    y = load_audio(path)
    y = librosa.util.fix_length(y, size=16000 * 3)
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
    return np.mean(mfcc, axis=1).reshape(1, -1)

# ---------------- UI — Input Section ----------------
st.subheader("🎧 Choose Input")

input_mode = st.radio("Select method:", ["📁 Upload Audio", "🎤 Record (browser)"])

temp_path = None

# Browser recorder support
recorder_available = False
try:
    from streamlit_audiorecorder import audiorecorder
    recorder_available = True
except:
    recorder_available = False

if input_mode == "📁 Upload Audio":
    uploaded = st.file_uploader("Upload audio file (wav/mp3/m4a/ogg)", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tf.write(uploaded.read())
        tf.close()
        temp_path = tf.name
        st.audio(temp_path)

else:
    if recorder_available:
        st.write("Click record:")
        audio_bytes = audiorecorder("Start Recording", "Stop")
        if audio_bytes:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tf.write(audio_bytes)
            tf.close()
            temp_path = tf.name
            st.audio(temp_path)
    else:
        st.warning("Browser recorder unavailable. Install: pip install streamlit-audiorecorder")
        st.info("Use Upload Audio instead.")

# ---------------- Predict Button ----------------
st.subheader("🔍 Run Accent & Age Detection")

if st.button("Detect Accent"):
    if not temp_path:
        st.error("Please upload or record audio first.")
    else:
        with st.spinner("Analyzing..."):

            features = extract_features(temp_path)

            # Transform using encoder
            try:
                X = enc.transform(features)
            except:
                X = features

            # Predict accent
            try:
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X)[0]
                    idx = np.argmax(probs)
                    predicted_label = enc.inverse_transform([idx])[0]
                    confidence = round(float(probs[idx] * 100), 2)
                else:
                    predicted_label = clf.predict(X)[0]
                    confidence = None
            except Exception as e:
                st.error("Model prediction error: " + str(e))
                st.stop()

            # Age estimate (simple)
            y = load_audio(temp_path)
            energy = float(np.mean(np.abs(y)))
            if energy < 0.1:
                age_group = "Senior"
            elif energy < 0.3:
                age_group = "Adult"
            else:
                age_group = "Young"

        # Display results
        st.success(f"🎯 **Accent:** {predicted_label}")
        if confidence is not None:
            st.info(f"Confidence: **{confidence}%**")
        st.write(f"**Estimated Age Group:** {age_group}")

        # Save to CSV
        log_prediction(os.path.basename(temp_path), predicted_label, confidence, age_group)
        st.success("📁 Saved to predictions_log.csv")

# ---------------- Show Last Logs ----------------
st.subheader("📄 Recent Predictions Log")

if os.path.exists(LOG_CSV):
    df = pd.read_csv(LOG_CSV)
    st.dataframe(df.tail(10))

# Footer
st.caption("Built by Renuka Renu — Accent Detective Project 🎧")
