import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from timm import create_model

# Emotion class labels (adjust if yours differ)
class_labels = ['angry', 'disgusted', 'frightened', 'happy', 'neutral', 'sad', 'surprised']
satisfaction_map = {
    'angry': 'dissatisfied',
    'disgusted': 'dissatisfied',
    'frightened': 'dissatisfied',
    'sad': 'dissatisfied',
    'happy': 'satisfied',
    'neutral': 'satisfied',
    'surprised': 'satisfied'
}

# Load ConvNeXt model
@st.cache_resource
def load_model():
    model = torch.load("FRconvnext_full(R)(A).pth", map_location='cpu', weights_only=False)
    model.eval()
    return model

# Preprocess uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

def crop_face(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return image  # No face detected, return original image

    x, y, w, h = faces[0]  # Only use the first detected face
    face = image_np[y:y+h, x:x+w]
    face_image = Image.fromarray(face)
    return face_image

# Predict function
def predict(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return class_labels[predicted_class], probs.squeeze().tolist()

# UI
st.set_page_config(page_title=" ESI w/ ConvNeXT ")
st.title("Employee Satisfaction Classifier")
st.write("Upload an image to classify emotions using a ConvNeXt model.")

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Webcam":
    camera_input = st.camera_input("Take a picture")
    if camera_input is not None:
        image = Image.open(camera_input).convert("RGB")

# If an image is provided from either method
if image:
    st.image(image, caption="Original Input", use_container_width=True)

    face_image = crop_face(image)
    st.image(face_image, caption="Cropped Face", use_container_width=False)

    model = load_model()
    input_tensor = preprocess_image(face_image)
    label, probabilities = predict(model, input_tensor)
    satisfaction = satisfaction_map[label]

    st.markdown(f"### You are feeling/looking: **{label}**")
    st.markdown(f"### You are probably: **{satisfaction}** while working")

    st.subheader("Confidence Scores")
    st.bar_chart({lbl: prob for lbl, prob in zip(class_labels, probabilities)})
