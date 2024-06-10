import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import librosa.display
import io

# Model ve scaler dosyalarını yükle
model = tf.keras.models.load_model('emotion_intensity_model.h5')
scaler = joblib.load('scaler.pkl')
encoder_emotion = joblib.load('encoder_emotion.pkl')
encoder_intensity = joblib.load('encoder_intensity.pkl')

# Özellikleri çıkarma fonksiyonu
def extract_features(data, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfccs.T

# Sliding window fonksiyonu
def sliding_window(data, window_size, step_size):
    num_segments = int((len(data) - window_size) / step_size) + 1
    segments = []
    for start in range(0, num_segments * step_size, step_size):
        end = start + window_size
        segments.append(data[start:end])
    return np.array(segments)

# Uygulama başlığı
st.title("Duygu ve Yoğunluk Tahmini")

# Ses dosyası yükleme
uploaded_file = st.file_uploader("Bir ses dosyası yükleyin", type=["wav", "mp3"])

if uploaded_file is not None:
    # Ses dosyasını yükleme
    duration = st.slider("Ses dosyasının süresini belirleyin (saniye)", min_value=1.0, max_value=10.0, value=2.5)
    offset = st.slider("Ses dosyasının başlangıç ofsetini belirleyin (saniye)", min_value=0.0, max_value=5.0, value=0.5)

    uploaded_file_object = io.BytesIO(uploaded_file.read())

    data, sample_rate = librosa.load(uploaded_file_object, sr=None, duration=duration, offset=offset)

    # Ses dosyasını oynatma
    st.audio(uploaded_file_object, format='audio/wav')

    # Ses dalga formu çizimi
    st.subheader("Ses Dalga Formu")
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(data, sr=sample_rate)
    st.pyplot(plt)

    # Spectrogram çizimi
    st.subheader("Spektrogram")
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 3))
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar()
    st.pyplot(plt)

    # Özellikleri çıkarma ve ölçekleme
    features = extract_features(data, sample_rate)
    window_size = 50
    step_size = 25
    feature_segments = sliding_window(features, window_size, step_size)

    feature_segments_scaled = np.zeros(
        (feature_segments.shape[0], feature_segments.shape[1], feature_segments.shape[2]))
    for i in range(feature_segments.shape[0]):
        feature_segments_scaled[i] = scaler.transform(feature_segments[i])

    # Model ile tahmin yapma
    y_pred_emotion, y_pred_intensity = model.predict(feature_segments_scaled)

    predicted_emotions = [encoder_emotion.categories_[0][np.argmax(pred)] for pred in y_pred_emotion]
    predicted_intensities = [encoder_intensity.categories_[0][np.argmax(pred)] for pred in y_pred_intensity]

    # Tahmin sonuçlarını zamanla ilişkilendirme
    segment_duration = (window_size / sample_rate) * step_size / window_size  # segment süresi saniye cinsinden
    timestamps = [(i * segment_duration) for i in range(len(predicted_emotions))]

    # Tahmin sonuçlarını gösterme
    st.subheader("Tahmin Sonuçları")

    results = [{"Time (s)": round(timestamps[i], 2), "Emotion": predicted_emotions[i], "Intensity": predicted_intensities[i]} for i in range(len(predicted_emotions))]
    st.write(results)

    # Zaman içinde tahmin grafiği çizme
    st.subheader("Zaman İçinde Tahminler")

    time_steps = feature_segments_scaled.shape[0]

    plt.figure(figsize=(14, 10))

    # Emotion predictions over time
    plt.subplot(2, 1, 1)
    for i, emotion in enumerate(encoder_emotion.categories_[0]):
        plt.plot(timestamps, [pred[i] for pred in y_pred_emotion], label=emotion)
    plt.title('Predicted Emotions Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion Probability')
    plt.legend()

    # Intensity predictions over time
    plt.subplot(2, 1, 2)
    for i, intensity in enumerate(encoder_intensity.categories_[0]):
        plt.plot(timestamps, [pred[i] for pred in y_pred_intensity], label=intensity)
    plt.title('Predicted Intensity Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity Probability')
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)
