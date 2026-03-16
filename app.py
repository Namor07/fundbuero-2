# =========================================
# Fundbüro-KI – YOLOv8 (neu)
# Schul- & Lernprojekt
# =========================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------------------
# Seiteneinstellungen
# -------------------------------
st.set_page_config(
    page_title="Fundbüro-KI",
    page_icon="🧳",
    layout="centered"
)

st.title("🧳 Fundbüro-KI")
st.write("Lade ein Bild eines gefundenen Gegenstands hoch.")

# -------------------------------
# YOLOv8 Modell laden (neueste Generation)
# -------------------------------
@st.cache_resource
def lade_modell():
    # YOLOv8n = aktuell, klein, stabil
    return YOLO("yolov8n.pt")

modell = lade_modell()

# -------------------------------
# Bild-Upload
# -------------------------------
bild_datei = st.file_uploader(
    "📷 Bild auswählen",
    type=["jpg", "jpeg", "png"]
)

if bild_datei is not None:
    # Bild laden
    bild = Image.open(bild_datei).convert("RGB")
    bild_np = np.array(bild)

    st.subheader("🖼️ Originalbild")
    st.image(bild, use_container_width=True)

    # -------------------------------
    # YOLO-Erkennung
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
    st.subheader("📋 Ergebnisse")

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
