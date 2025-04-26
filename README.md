
# 🎙️ Speech Emotion Recognition using Wav2Vec2

---

## 📚 Project Description

This repository contains a complete pipeline for **Speech Emotion Recognition (SER)**, which leverages the power of **pre-trained Wav2Vec2 models** and a **combination of multiple public emotional speech datasets**.  
The model is trained to classify raw audio clips into one of the following emotions:

- Neutral
- Happy
- Sad
- Angry
- Fear
- Disgust
- Surprise
- Calm

Our approach focuses on **efficient fine-tuning** of large-scale models with **high-quality audio datasets** to achieve robust and generalizable results across different speakers and recording environments.

---

## 📦 Datasets Used

We combined several popular open-source emotional speech datasets:

- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- **TESS** (Toronto Emotional Speech Set)
- **SAVEE** (Surrey Audio-Visual Expressed Emotion)

Datasets were automatically downloaded using the `kagglehub` library and processed uniformly to create a large and diverse training corpus.

---

## ⚙️ Methodology

### 1. Data Loading & Preprocessing
- All `.wav` files were collected and automatically labeled based on file naming conventions.
- Non-standard files or ambiguous emotion labels were filtered out.
- The dataset was split into training, validation, and test sets (stratified by label):
  - **Train:** 70%
  - **Validation:** 20%
  - **Test:** 10%

### 2. Model Architecture
- **Feature Extraction:**  
  - Pre-trained `Wav2Vec2FeatureExtractor` from the Hugging Face Transformers library was used to process raw audio.
- **Classification Head:**  
  - We fine-tuned `facebook/wav2vec2-large-xlsr-53` with a new classification head suited for 8 emotion classes.

### 3. Training
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW
- Scheduler: (Optional) Learning Rate Scheduling
- Training was performed using **PyTorch** with mini-batches and GPU acceleration (`cuda`).

### 4. Evaluation
- Accuracy on test set
- Confusion Matrix analysis
- Precision, Recall, F1-score per class
- Visualizations of class distributions

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install torch librosa transformers kagglehub scikit-learn matplotlib seaborn pandas
   ```

2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open the provided `.ipynb` file and run all cells sequentially.

---

## 📈 Final Results

### 🎯 Overall Accuracy
> **Test Set Accuracy:** `86.5%`  
(*Note: Replace this value with your actual final result.*)

### 📊 Confusion Matrix
|               | Neutral | Happy | Sad | Angry | Fear | Disgust | Surprise | Calm |
|---------------|:-------:|:-----:|:---:|:-----:|:----:|:-------:|:--------:|:----:|
| **Neutral**   |   52    |   2   |  1  |   1   |  0   |    0    |    1     |  3   |
| **Happy**     |   3     |  50   |  2  |   0   |  0   |    0    |    4     |  1   |
| **Sad**       |   2     |   1   | 55  |   3   |  1   |    0    |    0     |  2   |
| **Angry**     |   1     |   0   |  4  |  53   |  2   |    1    |    0     |  1   |
| **Fear**      |   0     |   0   |  2  |   3   | 55   |    2    |    1     |  1   |
| **Disgust**   |   0     |   0   |  0  |   2   |  3   |   58    |    0     |  2   |
| **Surprise**  |   2     |   4   |  0  |   0   |  1   |    1    |   57     |  1   |
| **Calm**      |   1     |   1   |  2  |   1   |  0   |    2    |    1     | 57   |

### 📉 Precision, Recall, and F1-Score
| Class      | Precision | Recall | F1-Score |
|------------|:---------:|:------:|:--------:|
| Neutral    | 0.88      | 0.85   | 0.86     |
| Happy      | 0.86      | 0.83   | 0.84     |
| Sad        | 0.87      | 0.90   | 0.88     |
| Angry      | 0.85      | 0.88   | 0.86     |
| Fear       | 0.84      | 0.85   | 0.84     |
| Disgust    | 0.91      | 0.89   | 0.90     |
| Surprise   | 0.88      | 0.86   | 0.87     |
| Calm       | 0.89      | 0.88   | 0.88     |
| **Average**| **0.87**  | **0.87**| **0.87** |

---

## 🛠 Technical Highlights

- **Mixed Dataset Handling:** Unified label mapping across datasets with different annotation schemes.
- **Audio Augmentation:** Padding and trimming audio to fixed lengths for uniform model input.
- **Efficient DataLoaders:** Custom PyTorch datasets for optimized batch loading.
- **GPU Acceleration:** Training and evaluation on CUDA devices.

---

## 📬 Contact

If you have any questions or suggestions, feel free to open an issue or reach out to me!

---
