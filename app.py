from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf

app = Flask(__name__)

# Modeli yükle
model = tf.keras.models.load_model('emotion_intensity_model.h5')


# Özellik çıkarma fonksiyonu
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


# Flask API tahmin endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['file']
    duration = float(request.form.get('duration', 2.5))
    offset = float(request.form.get('offset', 0.5))

    # Ses dosyasını yükle ve özellikleri çıkar
    data, sample_rate = librosa.load(audio_file, duration=duration, offset=offset)
    features = extract_features(data, sample_rate)

    # Sliding window
    window_size = 50
    step_size = 25
    feature_segments = sliding_window(features, window_size, step_size)

    # Model ile tahmin yap
    y_pred_emotion, y_pred_intensity = model.predict(feature_segments)

    # Tahminleri JSON formatında döndür
    return jsonify({
        'predicted_emotions': y_pred_emotion.tolist(),
        'predicted_intensities': y_pred_intensity.tolist()
    })


if __name__ == '__main__':
    app.run(debug=True)
