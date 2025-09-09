
**Speech Emotion Recognition**

**Project Overview**

This project uses deep learning to recognize human emotions (e.g., happy, sad, angry) from speech audio. It is a complete pipeline, from audio feature extraction to model training and evaluation. The entire project was developed on Google Colab to leverage free GPU resources.


**Key Features**

**Audio Feature Extraction:**
Uses Mel-Frequency Cepstral Coefficients (MFCCs) to convert raw audio into numerical features.

**Deep Learning Model:**
Employs a 1D Convolutional Neural Network (CNN) for classifying emotions.

**Dataset:** 
Trained and evaluated using the RAVDESS dataset.

**Real-world Testing:** 
Includes a script to test the trained model on new, unseen audio files.

Getting Started
Prerequisites
Google Account

Google Colab

Google Drive (for storing the dataset and model)

**Installation and Setup**
**Open the Project:**
Access the Jupyter notebook file (.ipynb) on Google Colab.

**Enable GPU:** 
Go to Runtime > Change runtime type and select GPU for faster training.

**Install Libraries:** 
The notebook includes a cell to install all required libraries (librosa, tensorflow, scikit-learn).

Usage
Follow the sequential steps in the provided Jupyter notebook.

**Data Preparation:**

Connect to your Google Drive and ensure the RAVDESS dataset is uploaded. The notebook will automatically find and organize the audio files.

**Feature Extraction:**

Run the script to extract MFCC features from all audio files.

**Model Training:** 
Train the 1D CNN model. The notebook will display training progress and accuracy.

**Evaluation:** 
Evaluate the model's performance on the test data using a classification report and a confusion matrix.

**Test a New Sample:**
Use the provided code to load your trained model and predict the emotion of a new .wav file.

**Results**

**The model's performance is detailed in the notebook's evaluation section. The key metrics include:**


Overall Accuracy: The percentage of correct predictions on the test set.

Per-Class Accuracy: The recall score for each individual emotion, showing how well the model identifies each class.

Confusion Matrix: A visual representation of correct vs. incorrect predictions.

**Future Improvements**
Use larger, more diverse datasets.

Explore advanced models like LSTMs or Transformers.

Implement data augmentation techniques to balance the dataset.

License
This project is licensed under the MIT License.

Author: Devamani
