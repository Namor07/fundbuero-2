import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------------------
# Seiteneinstellungen
# -------------------------------
st.set_page_config(
    page_title="Fundbüro-KI (YOLOv8 Online)",
    page_icon="🧳",
    layout="centered"
)

st.title("🧳 Fundbüro-KI")
st.write("Erkennung gefundener Gegenstände mit YOLOv8")

# -------------------------------
# YOLOv8 Modell laden (online)
# -------------------------------
@st.cache_resource
def lade_modell():
    # Lädt automatisch das vortrainierte YOLOv8s Modell von Ultralytics-Cloud
    return YOLO("yolov8s.pt")  # Ultralytics holt es automatisch online

modell = lade_modell()

# -------------------------------
# Bild-Upload
# -------------------------------
datei = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if datei:
    bild = Image.open(datei).convert("RGB")
    bild_np = np.array(bild)

    st.subheader("🖼️ Originalbild")
    st.image(bild, use_container_width=True)

    # -------------------------------
    # YOLOv8 Erkennung
    # -------------------------------
    ergebnisse = modell.predict(
        source=bild_np,
        conf=0.25,
        save=False
    )

    result = ergebnisse[0]

    # Annotiertes Bild
    annotiert = result.plot()

    st.subheader("📦 Erkannte Gegenstände")
    st.image(annotiert, use_container_width=True)

    # -------------------------------
    # Ergebnisliste
    # -------------------------------
    st.subheader("📋 Erkennungsergebnisse")

    if result.boxes is None or len(result.boxes) == 0:
        st.info("Keine Gegenstände erkannt.")
    else:
        for box in result.boxes:
            klasse_id = int(box.cls[0])
            klasse_name = result.names[klasse_id]
            sicherheit = float(box.conf[0])

            st.write(
                f"**Objekt:** {klasse_name}  \n"
                f"**Erkennungswahrscheinlichkeit:** {sicherheit:.1%}"
            )
