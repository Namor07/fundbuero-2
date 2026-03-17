import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# -----------------------------
# Seiteneinstellungen
# -----------------------------
st.set_page_config(
    page_title="KI-Fundbüro",
    page_icon="🧳",
    layout="centered"
)

st.title("🧳 KI-gestütztes Fundbüro")
st.write(
    "Lade ein Bild eines gefundenen Gegenstands hoch. "
    "Die KI erkennt automatisch Objekte im Bild."
)

# -----------------------------
# YOLOv8 Modell laden
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # kleines, schnelles Modell

model = load_model()

# -----------------------------
# Bild-Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📸 Bild eines gefundenen Gegenstands hochladen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Originalbild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("📷 Originalbild")
    st.image(image, use_column_width=True)

    # Bild in OpenCV-Format umwandeln
    img_array = np.array(image)

    # -----------------------------
    # Objekterkennung
    # -----------------------------
    with st.spinner("🔍 Bild wird analysiert ..."):
        results = model(img_array)

    result = results[0]

    # -----------------------------
    # Bounding Boxes zeichnen
    # -----------------------------
    annotated_img = img_array.copy()

    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Bounding Box zeichnen
        cv2.rectangle(
            annotated_img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(
            annotated_img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        detections.append({
            "Objekt": class_name,
            "Wahrscheinlichkeit": f"{confidence:.2%}"
        })

    # -----------------------------
    # Ergebnisbild anzeigen
    # -----------------------------
    st.subheader("🧠 Analyseergebnis")
    st.image(annotated_img, use_column_width=True)

    # -----------------------------
    # Erkennungsliste
    # -----------------------------
    st.subheader("📋 Erkannte Objekte")

    if detections:
        st.table(detections)
    else:
        st.info("Keine Objekte erkannt.")
