# =========================================
# Fundbüro-KI mit YOLOv5
# Schul- & Lernprojekt
# =========================================

import streamlit as st
import torch
import numpy as np
from PIL import Image

# -------------------------------
# Seiteneinstellungen
# -------------------------------
st.set_page_config(
    page_title="Fundbüro-KI (YOLOv5)",
    page_icon="🧳",
    layout="centered"
)

st.title("🧳 Fundbüro-KI")
st.write("Erkennung gefundener Gegenstände mit YOLOv5")

# -------------------------------
# YOLOv5 Modell laden
# -------------------------------
@st.cache_resource
def lade_modell():
    # YOLOv5s = klein, schnell, stabil
    modell = torch.hub.load(
        "ultralytics/yolov5",
        "yolov5s",
        pretrained=True
    )
    return modell

modell = lade_modell()

# -------------------------------
# Bild-Upload
# -------------------------------
datei = st.file_uploader(
    "📷 Bild hochladen",
    type=["jpg", "jpeg", "png"]
)

if datei:
    bild = Image.open(datei).convert("RGB")
    bild_np = np.array(bild)

    st.subheader("🖼️ Originalbild")
    st.image(bild, use_container_width=True)

    # -------------------------------
    # YOLOv5 Erkennung
    # -------------------------------
    ergebnisse = modell(bild_np)

    # Annotiertes Bild
    annotiertes_bild = ergebnisse.render()[0]

    st.subheader("📦 Erkannte Gegenstände")
    st.image(annotiertes_bild, use_container_width=True)

    # -------------------------------
    # Ergebnisliste
    # -------------------------------
    st.subheader("📋 Erkennungsergebnisse")

    daten = ergebnisse.pandas().xyxy[0]

    if daten.empty:
        st.info("Keine Gegenstände erkannt.")
    else:
        for _, zeile in daten.iterrows():
            st.write(
                f"**Objekt:** {zeile['name']}  \n"
                f"**Erkennungswahrscheinlichkeit:** {zeile['confidence']:.1%}"
            )
