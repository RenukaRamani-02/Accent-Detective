# Accent-Detective
🎙️ Accent Detective — Native Language Identification (HuBERT)
(B.Tech Project – Andhra Pradesh & Tamil Nadu Accent Classification)

Accent Detective is a machine-learning powered system that identifies the speaker’s native Indian language influence in English speech, such as:

Andhra Pradesh (Telugu English)

Tamil Nadu (Tamil English)

(You can add Kerala, Karnataka, North Indian etc.)

The system uses HuBERT (Self-supervised speech model) along with a custom classifier trained on regional Indian-English accent samples.
The project also includes age-group prediction, feature extraction, and CSV logging.

🚀 Features

✔ Upload audio file (WAV/MP3/M4A/FLAC/OGG)
✔ Optional: Record audio using Streamlit WebRTC
✔ Automatic accent classification
✔ Age group prediction
✔ HuBERT-based feature extraction
✔ Results saved automatically to predictions_log.csv
✔ Clean Streamlit web UI
✔ Works locally + deployable to Streamlit Cloud

🧠 Technology Stack
Component	Used For
Python	Main development
Streamlit	Web interface
HuBERT Base Model	Speech feature extraction
PyTorch	Model loading & inference
scikit-learn	Classification (Logistic Regression / SVM / RandomForest)
Librosa / Torchaudio	Audio processing
Pickle (.pkl)	Saved encoder & classifier
Pandas / CSV	Logging predictions
📂 Project Structure
Accent_Detective/
│
├── app.py
├── requirements.txt
├── AccentDetective_HuBERT_encoder.pkl
├── AccentDetective_HuBERT_model.pkl
├── labels.json              (optional)
├── predictions_log.csv      (auto-created)
└── README.md

⚙️ Installation (Local Machine)
1️⃣ Create Virtual Environment
python -m venv .venv

2️⃣ Activate
Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt


If torchaudio gives error:

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

▶️ Run the App
streamlit run app.py


Open in browser:

http://localhost:8501

📝 How to Use

Choose Record or Upload File

Speak a short English sentence (2–8 seconds)

Click Detect Accent & Age

See predictions

All predictions are saved in predictions_log.csv

📊 Logs Example (CSV)
timestamp,input_file,pred_label,confidence,age_group
2025-11-15 12:18:45,lithi.wav,Andhra Pradesh,98.6%,Adult

🧪 HuBERT Workflow (Simplified)

Load audio

Resample to 16 kHz

Extract HuBERT hidden states

Send to classifier

Predict accent

Predict age group

Log results

❤️ Author
Beera Renuka Ramani
B.Tech 
Accent Detective Project (2025)
