# Emotion Recognition from Speech 🎙️😄😭😠

This repository contains a Jupyter Notebook (`emotion-recognition.ipynb`) that demonstrates an end-to-end pipeline for recognizing emotions from speech audio using deep learning. The project leverages a hybrid Convolutional Neural Network (CNN) and Transformer architecture to classify audio into various emotional categories.

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Emotion Labels](#emotion-labels)
- [Results (Training & Validation)](#results-training--validation)
- [Future Work](#future-work)

## 📌 Project Overview

The goal of this project is to develop a robust model that can accurately identify human emotions from spoken words. It involves:

1. 🔊 **Audio Pre-processing:** Converting raw audio into Mel Spectrograms.
2. 🔁 **Data Augmentation:** Improving model generalization using various audio transformations.
3. 🧠 **Model Definition:** Combining CNNs + Transformers for effective learning.
4. 📈 **Training & Evaluation:** Training the model and validating its performance.

## ✨ Features

- 🎵 **Mel Spectrogram Extraction**: Converts waveforms into log-Mel spectrograms.
- 🔄 **Audio Augmentation**: Includes:
  - 🔊 Random volume gain
  - 🔇 Frequency & time masking
  - 📈 Gaussian noise
  - 🎚️ Pitch shifting
  - 🔃 Polarity inversion
  - ✂️ Random cropping/padding
- 🧩 **Hybrid CNN-Transformer Model (ACNT)**
  - 🧱 CNN backbone with residual connections
  - 🕒 Positional encoding for temporal context
  - 🧠 Transformer encoder for long-range dependencies
  - 🎯 Classification head for emotion labels
- ⚙️ **PyTorch Implementation**
- 🚀 **Efficient Data Loading** with custom Dataset & DataLoader

## 🎤 Dataset

Using the RAVDESS dataset:
- Speech & song recordings
- Actors expressing **8 emotions**:
  - 😐 neutral
  - 😌 calm
  - 😊 happy
  - 😢 sad
  - 😡 angry
  - 😱 fearful
  - 🤢 disgust
  - 😲 surprised

🗂️ Dataset Location: `/kaggle/input/ravdess-emotional-speech-audio`

🔀 Split:
- Train: 70%
- Validation: 15%
- Test: 15% (via 0.5 split of temp 20%)

## 🏗️ Model Architecture

### 🔁 PositionalEncoding
Uses sine and cosine functions to encode sequence order.

### 🔄 ResidualBlock
Improves gradient flow and deep CNN training.

### 🧠 ACNT (Audio CNN-Transformer Network)

**Key Parameters:**
- `num_classes`: 🎯 Number of emotion categories
- `in_channel`: 🔈 Spectrogram input channel (1 for mono)
- `input_shape`: 📐 Dimensions (e.g., 40x174)
- `cnn_channels`: 🧱 [16, 32, 64, 128]
- `transformer_dim`, `num_heads`, `num_layers`: 🧠 Transformer specs
- `pe_max_len`, `dropout_rate`, `use_transformer`: ⚙️ Flexibility controls

**Architecture Flow:**
1. 🧱 CNN layers with BatchNorm, ReLU, Dropout, MaxPool
2. 🔀 Reshape CNN features and project linearly
3. 🕒 Positional encoding (if enabled)
4. 🧠 Transformer Encoder for temporal dependencies
5. 📏 Global average pooling
6. 🎯 Classification head

✅ **Weight Initialization:**
- Kaiming for conv layers
- Xavier for linear layers

## ⚙️ Setup and Installation

Install dependencies:
```bash
!pip install torchmetrics
!sudo apt update
!sudo apt install sox libsox-fmt-all
!pip install sox
```

📦 Use Python 3.x. Designed for Kaggle GPU; works with local setups too.

## ▶️ Usage

1. 🔽 Clone repository:
```bash
git clone <repository_url>
cd <repository_directory>
```
2. 📥 Download RAVDESS dataset into `/kaggle/input/ravdess-emotional-speech-audio`
3. 🧪 Run notebook:
```bash
jupyter notebook emotion-recognition.ipynb
```

🏃 The notebook will:
- Load modules
- Extract Mel spectrograms
- Apply augmentations
- Split data
- Initialize ACNT model
- Setup loss, optimizer, scheduler
- Train & validate the model

## 🎭 Emotion Labels

```python
Emotion_labels = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

emotion_to_idx = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}
```

## 📊 Results (Training & Validation)

The training loop prints:
- 📉 Training Loss
- 📉 Validation Loss
- 🎯 Accuracy on both
- 🔁 Scheduler adapts learning rate

📈 MEL spectrogram visualizations included.

## 🔮 Future Work

- 🎯 Hyperparameter tuning (grid/Bayesian search)
- 🔁 K-fold cross-validation
- 🎛️ Advanced augmentations (e.g., reverberation)
- 🗃️ Larger, diverse datasets
- 🌐 Model deployment (web/mobile)
- 🧪 Model optimization (quantization/pruning)

---
**Made with ❤️ for Speech Emotion Recognition**
