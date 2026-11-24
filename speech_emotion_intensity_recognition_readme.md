# Speech Emotion & Intensity Recognition (Multi-Task BiLSTM)

A deep learning project that simultaneously classifies **emotional states** and **emotional intensity** from human speech using a Multi-Task Bidirectional Long Short-Term Memory (**BiLSTM**) neural network.

## 📌 Project Overview

Speech emotion recognition (SER) is a key component in human-computer interaction. Beyond identifying *what* emotion is being expressed, understanding *how strongly* it is expressed adds crucial context. 

This project implements a multi-output deep learning model trained on the **RAVDESS** dataset to predict two targets concurrently from raw audio segments:
1. **Emotion Class (8 classes):** Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Surprise.
2. **Emotional Intensity (2 classes):** Normal, Strong.

---

## 📊 Dataset

The model is trained on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset (specifically the `audio_speech_actors_01-24` subset containing 1,440 audio files across 24 actors).

* **Target Parsing:** Labels are extracted directly from the RAVDESS filename convention (`part[2]` for emotion, `part[3]` for intensity).
* **Exploratory Data Analysis (EDA):** Includes class distribution visualizations, audio waveforms (`waveplot`), and Short-Time Fourier Transform (`STFT`) spectrograms.

---

## ⚙️ Audio Preprocessing & Feature Extraction

1. **Audio Loading:** Each audio file is loaded with `librosa` using a fixed `duration=2.5` seconds and `offset=0.5` seconds to trim leading silence.
2. **Feature Extraction:** 13 Mel-Frequency Cepstral Coefficients (**MFCCs**) are extracted per frame (`n_mfcc=13`).
3. **Sliding Window Segmentation:** To capture temporal dynamics and augment training samples, extracted MFCC sequences are segmented using a sliding window approach:
   * `window_size = 50`
   * `step_size = 25`
4. **Data Augmentation Utilities:** Helper functions for Gaussian noise injection, time-stretching, pitch-shifting, and time-shifting are implemented in the pipeline.
5. **Normalization & Encoding:** 
   * Features are standardized per sample using `StandardScaler`.
   * Target labels are encoded using `OneHotEncoder`.
   * Data is split into **70% Training** and **30% Validation/Test** sets (`random_state=21`).

---

## 🧠 Model Architecture

The model uses the Keras Functional API to build a shared sequential feature extractor branching into two specialized classification heads:

```text
Input (time_steps=50, features=13)
   │
   ▼
Bidirectional LSTM (128 units, return_sequences=True)
   │
   ▼
Dropout (0.5)
   │
   ▼
Bidirectional LSTM (64 units, return_sequences=False)
   │
   ▼
Dropout (0.5)
   │
   ▼
Dense (32 units, ReLU)
   │
   ▼
Dropout (0.5)
   ├───────────────────────────┐
   ▼                           ▼
Emotion Output Head       Intensity Output Head
Dense (8, Softmax)        Dense (2, Softmax)
```

---

## 🚀 Training Configuration

* **Optimizer:** Adam
* **Loss Function:** `categorical_crossentropy` (applied to both `emotion_output` and `intensity_output`)
* **Metrics:** Accuracy for both output heads
* **Batch Size & Epochs:** `batch_size=32`, `epochs=100`
* **Callbacks:**
  * `EarlyStopping`: Monitors `val_loss` with `patience=10` and restores the best weights.
  * `ReduceLROnPlateau`: Monitors `val_loss`, reducing learning rate by `factor=0.5` after `patience=5` (`min_lr=5e-4`).

---

## 🛠️ Tech Stack & Dependencies

* **Python 3.x**
* **Deep Learning:** TensorFlow / Keras
* **Audio Processing:** Librosa
* **Data Manipulation & ML:** NumPy, Pandas, Scikit-learn
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Repository Structure

```text
├── duygu ve yoğunluk tanıma.ipynb   # Main Jupyter Notebook (EDA, preprocessing, training, evaluation)
├── saved_model/                    # Trained Keras model files
└── README.md                       # Project documentation
```

---

## 💻 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Adil-Can-Kural/DuyguTanima.git
   cd DuyguTanima
   ```

2. **Install required libraries:**
   ```bash
   pip install tensorflow librosa numpy pandas scikit-learn matplotlib seaborn
   ```

3. **Dataset Setup:**
   Download the RAVDESS Emotional Speech Audio dataset and update the `Ravdess` directory path in the notebook to point to your local `audio_speech_actors_01-24/` folder.

4. **Run the Notebook:**
   Open `duygu ve yoğunluk tanıma.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run all cells to reproduce the training and evaluation plots.