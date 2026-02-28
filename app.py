# =================================================================================
#  CAT AND DOG CLASSIFIER(CNN)
# =================================================================================

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


model = load_model("model.h5")

st.title("CAT AND DOG CLASSFIER(CNN)")

IMG_SIZE = 256

st.write("Upload an image of a cat or dog")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0   # normalize if training used rescale=1./255
    image = np.expand_dims(image, axis=0)
    return image



if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image",width="stretch" )


    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    confidence = float(prediction)

    if confidence > 0.5:
        st.success(f"Prediction: 🐶 Dog ({confidence*100:.2f}% confidence)")
    else:
        st.success(f"Prediction: 🐱 Cat ({(1-confidence)*100:.2f}% confidence)")


