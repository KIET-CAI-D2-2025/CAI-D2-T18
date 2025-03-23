import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten

# Define the path to the CREMA-D dataset
DATASET_PATH = "AudioWav/"

# Define emotion mapping based on CREMA-D naming convention
emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# Function to extract MFCC features from audio files
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
        print(f"Error processing {file_path}: {e}")
        return None

# Function to load data from the CREMA-D dataset
def load_data(data_path):
    features, labels = [], []
    for file in tqdm(os.listdir(data_path)):
        if file.endswith(".wav"):
            parts = file.split("_")  # Example: 1001_IEO_HAP_HI.wav
            emotion = parts[2]  # Extract emotion code
            if emotion in emotion_map:
                file_path = os.path.join(data_path, file)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion_map[emotion])  # Convert code to emotion name
    return np.array(features), np.array(labels)

# Load dataset
print("Loading dataset...")
X, y = load_data(DATASET_PATH)
print(f"Loaded {len(X)} audio samples.")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape data for the model
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Neural Network model
model = Sequential([
    Flatten(input_shape=(40, 100, 1)),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(emotion_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("speech_emotion_model1.h5")
print("Model saved successfully!")

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Function to predict emotion from a new audio file
def predict_emotion(file_path):
    feature = extract_features(file_path)
    if feature is not None:
        feature = feature.reshape(1, 40, 100, 1)
        prediction = model.predict(feature)
        emotion = label_encoder.inverse_transform([np.argmax(prediction)])
        return emotion[0]
    return "Unknown"

# Example usage (Replace with your test audio file)
test_audio = "audio.wav"
print(f"Predicted Emotion: {predict_emotion(test_audio)}")
