# **Emotion Recognition from Speech**

This repository contains a Jupyter Notebook (emotion-recognition.ipynb) that demonstrates an end-to-end pipeline for recognizing emotions from speech audio using deep learning. The project leverages a hybrid Convolutional Neural Network (CNN) and Transformer architecture to classify audio into various emotional categories.

## **Table of Contents**

* [Project Overview](#bookmark=id.tyinxw5m8t5p)  
* [Features](#bookmark=id.9j3w9cyivtq)  
* [Dataset](#bookmark=id.erzrv6dzweso)  
* [Model Architecture](#bookmark=id.k3dln9r76q20)  
* [Setup and Installation](#bookmark=id.1wicpfyx4sz)  
* [Usage](#bookmark=id.xm9cvai8r6x)  
* [Emotion Labels](#bookmark=id.4ozd9d3txmqj)  
* [Results (Training & Validation)](#bookmark=id.ov0wc9jmwvdz)  
* [Future Work](#bookmark=id.z31zjkr30clf)

## **Project Overview**

***The goal of this project is to develop a robust model that can accurately identify human emotions from spoken words.*** It involves:

1. **Audio Pre-processing:** Converting raw audio into suitable features (Mel Spectrograms).  
2. **Data Augmentation:** Applying various augmentation techniques to improve model generalization.  
3. **Model Definition:** Designing a hybrid deep learning model combining CNNs for local feature extraction and Transformers for capturing temporal dependencies.  
4. **Training & Evaluation:** Training the model on a labeled dataset and evaluating its performance.

## **Features**

* **Mel Spectrogram Extraction:** Converts audio waveforms into log-Mel spectrograms, which are effective features for audio analysis.  
* **Comprehensive Audio Augmentation:** Includes:  
  * Random volume gain.  
  * Frequency and time masking (Spectrogram Augmentation).  
  * Adding Gaussian noise.  
  * Random pitch shifting.  
  * Random polarity inversion.  
  * Random cropping and padding to a fixed length.  
* **Hybrid CNN-Transformer Model (ACNT):**  
  * CNN backbone with residual connections for hierarchical feature learning.  
  * Positional Encoding to maintain temporal information.  
  * Transformer Encoder for capturing long-range dependencies across time frames.  
  * Classification head for predicting emotion labels.  
* **PyTorch Implementation:** All components are built using PyTorch for flexibility and performance.  
* **Data Loading Pipeline:** Custom Dataset and DataLoader for efficient batch processing.

## **Dataset**

The model is trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.  
The dataset includes:

* Speech and song recordings.  
* Actors expressing 8 different emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised.  
* Each file name contains information about the emotion, which is parsed for labeling.

Data Directory:  
The notebook expects the RAVDESS dataset to be located at:  
/kaggle/input/ravdess-emotional-speech-audio  
Dataset Split:  
The dataset is split into:

* **Training:** 70%  
* **Validation:** 15%  
* **Test:** 15% (Note: The notebook currently uses X\_temp\_paths and y\_temp for both validation and test sets in the train\_test\_split, effectively splitting the original 20% into 15% validation and 5% test from the full dataset. For a standard 70/15/15 split, the test\_size for the second split should be adjusted to 0.5 of X\_temp\_paths and y\_temp).

## **Model Architecture**

The ACNT (Audio CNN-Transformer) model is designed to process 2D Mel spectrograms and classify emotions.

### **PositionalEncoding**

A standard positional encoding layer using sine and cosine functions to inject information about the relative or absolute position of elements in the sequence. This is crucial for the Transformer to understand the temporal order of features.

### **ResidualBlock**

A simple residual block used within the CNN part of the ACNT model. It adds the input to the output of a sublayer if their dimensions match, helping to prevent vanishing gradients and improve training of deeper networks.

### **ACNT (Audio CNN-Transformer Network)**

This is the main model class.

**Initialization Parameters:**

* num\_classes: Number of emotion categories (default: 8).  
* in\_channel: Input channel for the spectrogram (usually 1 for mono audio).  
* input\_shape: Expected input dimensions of the Mel spectrogram (e.g., (40, 174)).  
* cnn\_channels: List of output channels for each CNN block (e.g., \[16, 32, 64, 128\]).  
* transformer\_dim: Dimensionality of the Transformer's input and output.  
* num\_heads: Number of attention heads in the Transformer.  
* num\_layers: Number of Transformer Encoder layers.  
* pe\_max\_len: Maximum sequence length for positional encoding.  
* dropout\_rate: Dropout rate applied in various layers.  
* use\_transformer: Boolean flag to enable/disable the Transformer part (useful for ablation studies).

**Architecture Flow:**

1. **CNN Backbone:**  
   * Composed of multiple Conv2d, BatchNorm2d, LeakyReLU (or ReLU), Dropout2d, and MaxPool2d layers.  
   * Includes ResidualBlocks for enhanced feature learning.  
   * Transforms the input Mel spectrogram \[B, 1, 40, 174\] into a higher-level feature map (e.g., \[B, 128, 5, 21\]).  
2. **Feature Flattening and Projection:**  
   * The CNN output is permuted and reshaped to \[B, W, H \* C\], where W is the number of time frames, H is the number of Mel bins, and C is the number of channels.  
   * A Linear layer self.proj projects this flattened feature into the transformer\_dim.  
3. **Positional Encoding:**  
   * If use\_transformer is True, positional encodings are added to the projected CNN features.  
4. **Transformer Encoder:**  
   * Comprises num\_layers of TransformerEncoderLayers.  
   * Processes the sequential features, applying self-attention to capture temporal dependencies.  
5. **Global Average Pooling:**  
   * The output from the Transformer (or projected CNN features if Transformer is disabled) is subjected to mean pooling across the time dimension to get a fixed-size representation.  
6. **Classification Head:**  
   * A simple Sequential block with Linear, LeakyReLU, and Dropout layers.  
   * Outputs the raw logits for num\_classes emotions.

Weight Initialization:  
The model uses kaiming\_normal\_ for convolutional layers (suitable for Leaky ReLU) and xavier\_normal\_ for linear layers, with biases initialized to zero.

## **Setup and Installation**

To run this notebook, you need to install the following Python libraries:

\!pip install torchmetrics  
\!sudo apt update  
\!sudo apt install sox libsox-fmt-all  
\!pip install sox

Ensure you have a suitable environment (e.g., Anaconda, Miniconda) with Python 3.x installed. The notebook is designed to run on a Kaggle GPU environment but can be adapted for local GPU/CPU setups.

## **Usage**

1. **Clone the Repository (or download the notebook):**  
   git clone \<repository\_url\>  
   cd \<repository\_directory\>

   (Replace \<repository\_url\> and \<repository\_directory\> with actual values if this is part of a larger project.)  
2. Download the RAVDESS dataset:  
   The notebook assumes the RAVDESS dataset is available at /kaggle/input/ravdess-emotional-speech-audio. You will need to download the dataset and place it in the appropriate directory or adjust the data\_dir variable in the notebook.  
3. Run the Jupyter Notebook:  
   Open the emotion-recognition.ipynb file using Jupyter Lab or Jupyter Notebook:  
   jupyter notebook emotion-recognition.ipynb

   Execute the cells sequentially.

The notebook will:

* Import necessary modules.  
* Load and preprocess the audio data, extracting Mel spectrogram features.  
* Apply various data augmentation techniques during dataset loading.  
* Split the data into training, validation, and test sets.  
* Define and initialize the ACNT model.  
* Set up the loss function (CrossEntropyLoss with label smoothing), optimizer (AdamW), and learning rate scheduler (ReduceLROnPlateau).  
* Proceed with the training and validation loops.

## **Emotion Labels**

The following mapping is used for emotion codes to labels:

Emotion\_labels \= {  
    '01': 'neutral',  
    '02': 'calm',  
    '03': 'happy',  
    '04': 'sad',  
    '05': 'angry',  
    '06': 'fearful',  
    '07': 'disgust',  
    '08': 'surprised'  
}

The numerical encoding used in the dataset is:

emotion\_to\_idx \= {  
    'neutral': 0,  
    'calm': 1,  
    'happy': 2,  
    'sad': 3,  
    'angry': 4,  
    'fearful': 5,  
    'disgust': 6,  
    'surprised': 7  
}

## **Results (Training & Validation)**

The notebook includes a training loop that tracks and prints training and validation loss and accuracy over epochs. A sample output shows:

* **Epoch Loss:** The training loss decreases over epochs.  
* **Validation Loss:** The validation loss also decreases, indicating the model is learning and generalizing.  
* **Training Accuracy:** The model's accuracy on the training set improves.  
* **Validation Accuracy:** The model's accuracy on unseen validation data improves, demonstrating generalization.

The ReduceLROnPlateau scheduler will adjust the learning rate if the validation loss stops improving.

A MEL spectrogram plot example is also provided in the notebook for visualization of the extracted features.

## **Future Work**

* **Hyperparameter Tuning:** Further optimize model hyperparameters (learning rate, dropout, CNN channels, transformer dimensions, etc.) using techniques like Grid Search or Bayesian Optimization.  
* **Cross-Validation:** Implement k-fold cross-validation for more robust evaluation.  
* **More Advanced Augmentations:** Explore other audio augmentation techniques (e.g., reverberation, shifting, changing pitch/speed with more advanced methods).  
* **Larger Datasets:** Evaluate the model on larger and more diverse speech emotion datasets.  
* **Deployment:** Consider deploying the trained model as a web service or a mobile application for real-time emotion recognition.  
* **Quantization/Pruning:** Optimize the model for deployment by applying quantization or pruning techniques.