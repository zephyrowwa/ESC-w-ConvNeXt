import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from timm import create_model
from huggingface_hub import hf_hub_download
from PIL import Image

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

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="zephyrowwa/convnxtferhehe", filename="FRconvnext_full(R)(A).pth")
    model = torch.load(model_path, map_location="cpu", weights_only= False)
    model.eval()
    return model

def preprocess_face(face_img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    pil_face = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return transform(pil_face).unsqueeze(0)

def detect_and_classify_faces(frame, model, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        input_tensor = preprocess_face(face_img)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            emotion = class_labels[pred_idx]
            satisfaction = satisfaction_map[emotion]

        label = f"{emotion} ({satisfaction})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

st.set_page_config(page_title="RT emotionengine")
st.title("RT Facial Emotion Classifier")
st.markdown("Haar Cascade + ConvNeXt 2 classify yo emotion yuh")

run = st.checkbox("start the emotion classifyin' ")
FRAME_WINDOW = st.image([])

model = load_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = None

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("walang webcam pre ano ba yan.")
            break

        frame = cv2.flip(frame, 1)
        annotated_frame = detect_and_classify_faces(frame, model, face_cascade)

        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
else:
    if cap:
        cap.release()
    st.write("tick da box to strat.")