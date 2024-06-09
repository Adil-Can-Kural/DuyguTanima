import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model


# Modeli yükle
@st.cache(allow_output_mutation=True)
def load_model_from_file():
    model = load_model('path_to_your_model.h5')
    return model


model = load_model_from_file()


# Özellik çıkarma fonksiyonları
def extract_features(data, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs.T


def sliding_window(data, window_size, step_size):
    num_segments = int((len(data) - window_size) / step_size) + 1
    segments = []
    for start in range(0, num_segments * step_size, step_size):
        end = start + window_size
        segments.append(data[start:end])
    return np.array(segments)


def preprocess_audio(file):
    data, sample_rate = librosa.load(file, sr=None)
    features = extract_features(data, sample_rate)
    feature_segments = sliding_window(features, window_size=50, step_size=25)

    scaler = StandardScaler()
    feature_segments_scaled = np.zeros(
        (feature_segments.shape[0], feature_segments.shape[1], feature_segments.shape[2]))
    for i in range(feature_segments.shape[0]):
        feature_segments_scaled[i] = scaler.fit_transform(feature_segments[i])

    return feature_segments_scaled


# Streamlit uygulaması
st.title('Duygu Tanıma Uygulaması')

uploaded_file = st.file_uploader("Bir ses dosyası yükleyin", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    features = preprocess_audio(uploaded_file)

    y_pred_emotion, y_pred_intensity = model.predict(features)

    st.write("Duygu Tahminleri:")
    st.write(y_pred_emotion)
    st.write("Yoğunluk Tahminleri:")
    st.write(y_pred_intensity)
