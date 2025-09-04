import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your trained model
model = load_model("pneumonia_model.keras")

# Prediction function
def predict_xray(img):
    img = img.resize((150, 150))  # same size as training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return {"PNEUMONIA": float(prediction), "NORMAL": float(1 - prediction)}
    else:
        return {"NORMAL": float(1 - prediction), "PNEUMONIA": float(prediction)}

# Gradio interface
iface = gr.Interface(
    fn=predict_xray,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Chest X-ray Pneumonia Classifier",
    description="Upload a chest X-ray to classify as NORMAL or PNEUMONIA."
)

if __name__ == "__main__":
    iface.launch()
