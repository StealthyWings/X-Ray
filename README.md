# Chest X-Ray Pneumonia Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify chest X-ray images as Normal or Pneumonia.
The model is trained on [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and deployed using Gradio on Hugging Face Spaces

<img width="1863" height="828" alt="image" src="https://github.com/user-attachments/assets/6777f327-dbfb-4caf-888f-f8729ff4bf8c" />

# Live Demo:

<https://huggingface.co/spaces/StealthyWings/XRay-Pneumonia>

## Features:

- Upload a chest X-ray image and get instant predictions: Normal or Pneumonia
- Data augmentation for better generalization
- CNN model with convolutional, pooling, dropout, and dense layers
- Evaluation with accuracy, precision, recall, and confusion matrix

## Model Architecture:

- **Input:** 150x150 RGB chest X-ray images
- **Conv2D + MaxPooling2D:** Extract spatial features
- **Flatten + Dense Layers:** Learn decision boundaries
- **Dropout:** Reduce overfitting
- **Output:** Sigmoid neuron -> binary classification (Normal vs Pneumonia)

## Training:

- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 10
- Metrics: Accuracy, Precision, Recall

## Results:

- Achieved strong performance on the test set
- Visualizations: training vs validation accuracy & loss plots
- Confusion matrix for detailed evaluation

# Usage:

### Local Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/StealthyWings/X-Ray.git
   cd X-Ray

2. Install dependencies:
  **pip install -r requirements.txt**

3. Run the app:
  **python app.py**

4. Open the local Gradio link in your browser and upload X-ray images.

# License

This project is for educational purposes.



