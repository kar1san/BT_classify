import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Define the preprocess_image function
def preprocess_image(image):
    img = Image.open(image).resize((200, 200))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('model.h5')

model = load_trained_model()

# Define the tumor classes and their descriptions
tumor_classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
tumor_descriptions = {
    'Glioma Tumor': 'Gliomas are tumors that originate in the glial cells of the brain or spine. They vary in aggressiveness and can cause headaches, seizures, and memory loss.',
    'Meningioma Tumor': 'Meningiomas are usually benign tumors that develop from the meninges. They can cause headaches, vision problems, and seizures due to their size or location.',
    'No Tumor': 'No signs of a brain tumor are detected in the MRI image. The brain appears normal, but further medical evaluation may be necessary.',
    'Pituitary Tumor': 'Pituitary tumors develop in the pituitary gland and can affect hormone production, leading to symptoms like vision problems, headaches, and hormonal imbalances.'
}

# Custom CSS for background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F0F2F6; /* Light grey background */
        color: #333333; /* Dark grey text */
        font-family: 'Arial', sans-serif; /* Font style */
    }
    .css-18ni7ap {
        background-color: #4B8BBE; /* Blue header */
        color: #FFFFFF; /* White text */
    }
    .stButton>button {
        background-color: #4B8BBE; /* Blue button */
        color: white;
    }
    .stMarkdown {
        margin-bottom: 30px;
    }
    .prediction-table {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the UI
st.title('🧠 Brain Tumor Classification')
st.write("""
    **Upload an MRI image of a brain, and the model will classify it into one of the following categories:**
    - Glioma Tumor
    - Meningioma Tumor
    - No Tumor
    - Pituitary Tumor
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner('Processing...'):
        image = preprocess_image(uploaded_file)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        predicted_class_name = tumor_classes[predicted_class]

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write(f"**Image Name:** {uploaded_file.name}")
        st.write(f"**Predicted Class:** {predicted_class_name}")
        st.write(f"**Confidence:** {prediction[0][predicted_class]:.4f}")

        st.write("### Prediction Confidence for All Classes:", unsafe_allow_html=True)
        confidence_data = {"Class": tumor_classes, "Confidence": [f"{conf:.4f}" for conf in prediction[0]]}
        st.table(confidence_data)

        st.write(f"### Description of Predicted Class: {predicted_class_name}")
        st.write(tumor_descriptions[predicted_class_name])
        
        st.success('Classification completed!')
else:
    st.write("Please upload an MRI image file to get a classification.")

# Additional context and user engagement
with st.expander("Read More"):
    st.markdown("""
        ## About This Website 🌐
        This brain tumor classification application leverages deep learning techniques to provide insights into MRI images of the brain. 
        It aims to demonstrate the capabilities of machine learning in the field of medical imaging. The model used here has been trained on a dataset of MRI images 
        and is designed to classify images into four categories: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor.

        ## Usage Instructions
        - **Upload an Image**: Use the file uploader to select an MRI image in JPG, JPEG, or PNG format.
        - **View Predictions**: Once the image is uploaded, the website will process it and display the predicted class along with the confidence levels for each category.
        - **Educational Purpose**: Please note that this tool is intended for educational and demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
        ## Recommendations
        After obtaining the classification results, it is important to consult with a medical professional for a comprehensive analysis and further steps.
        ## Contact 📨
        If you have any questions, suggestions, or feedback, feel free to reach out to the project maintainers at [admin123@gmail.com].

        Thank you for using our Brain Tumor Classification website!
    """)
