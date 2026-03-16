"""
Fundbüro-App mit YOLOv8
Autor: Schul-/Lernprojekt
Beschreibung:
Diese App erkennt gefundene Gegenstände auf Bildern mithilfe von YOLOv8.
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

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
# YOLOv8 Modell laden
# -------------------------------
@st.cache_resource
def lade_modell():
    return YOLO("yolov8n.pt")  # kleines, schnelles Modell

modell = lade_modell()

# -------------------------------
# Bild-Upload
# -------------------------------
datei = st.file_uploader(
    "📷 Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if datei:
    # Bild öffnen
    bild = Image.open(datei).convert("RGB")
    bild_array = np.array(bild)

    st.subheader("🖼️ Originalbild")
    st.image(bild, use_container_width=True)

    # -------------------------------
    # YOLO-Analyse
    # -------------------------------
    ergebnis = modell(bild_array)[0]

    annotiertes_bild = ergebnis.plot()
    annotiertes_bild = cv2.cvtColor(annotiertes_bild, cv2.COLOR_BGR2RGB)

    st.subheader("📦 Erkannte Gegenstände")
    st.image(annotiertes_bild, use_container_width=True)

    # -------------------------------
    # Ergebnisliste
    # -------------------------------
    st.subheader("📋 Erkennungsergebnisse")

    if len(ergebnis.boxes) == 0:
        st.info("Keine Gegenstände erkannt.")
    else:
        for box in ergebnis.boxes:
            klasse = ergebnis.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            st.write(
                f"**Objekt:** {klasse}  \n"
                f"**Erkennungswahrscheinlichkeit:** {confidence:.2%}"
            )
