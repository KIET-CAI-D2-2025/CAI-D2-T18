import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("speech_emotion_model.h5")

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
emotion_colors = {
    "Angry": "#FF5733",
    "Disgust": "#8B4513",
    "Fear": "#800080",
    "Happy": "#FFD700",
    "Neutral": "#4682B4",
    "Sad": "#1E90FF"
}

# Streamlit UI
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .big-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #FF4500;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #333333;
        }
        .emotion-box {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-title">🎤 Speech Emotion Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an audio file and predict its emotion!</p>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=100):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# Predict emotion function
def predict_emotion(audio_file):
    feature = extract_features(audio_file)
    if feature is not None:
        feature = feature.reshape(1, 40, 100, 1)
        prediction = model.predict(feature)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        return predicted_emotion
    return None

# Process uploaded file
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict emotion
    predicted_emotion = predict_emotion("temp.wav")

    if predicted_emotion:
        # Display emotion with color
        emotion_color = emotion_colors[predicted_emotion]
        st.markdown(
            f'<div class="emotion-box" style="background-color: {emotion_color}; color: white;">'
            f"Predicted Emotion: {predicted_emotion}</div>",
            unsafe_allow_html=True
        )
