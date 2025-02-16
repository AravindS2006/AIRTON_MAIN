import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

# Load the model (replace with your actual model loading)
def load_model():
    try:
        model = tf.keras.models.load_model("my_model2.h5")  # Assuming in same directory
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.exception(e) # Display the full traceback in the Streamlit app
        return None

model = load_model()

# --- Image Processing and Prediction ---
def predict_glaucoma(image_data):
    if model is None:
        st.error("Model not loaded. Please check the logs.")
        return "Error: Model not loaded", 0.0

    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_data))

        # Process the image
        image = ImageOps.fit(image, (100, 100), Image.LANCZOS) # Changed here!
        image = image.convert('RGB')
        image = np.asarray(image)
        image = image.astype(np.float32) / 255.0
        img_reshape = image[np.newaxis, ...]

        # Predict using the model
        prediction = model.predict(img_reshape)
        pred = prediction[0][0]

        # Determine classification
        if pred > 0.5:
            result = "Healthy"
        else:
            result = "Affected by Glaucoma"

        return result, pred

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error during prediction", 0.0

# --- Streamlit App ---
st.title("Glaucoma Detection")

uploaded_file = st.file_uploader("Upload a fundus image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            image_bytes = uploaded_file.read()  # Read the image as bytes

            diagnosis, confidence = predict_glaucoma(image_bytes)

        st.write("Diagnosis:", diagnosis)
        st.write("Confidence:", confidence)