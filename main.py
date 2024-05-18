import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Define the preprocess_image function
def preprocess_image(image):
    # Open the uploaded image
    img = Image.open(image)
    
    # Resize the image to 200x200 pixels
    img_resized = img.resize((200, 200))
    
    # Convert the image to a numpy array
    img_array = np.array(img_resized) / 255.0  # Normalize pixel values
    
    # Reshape the array to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Load the trained model
@st.cache_resource
def load_trained_model():
    return load_model('model.h5')  # Replace 'model.h5' with the path to your saved model

model = load_trained_model()

# Define the tumor classes
tumor_classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Custom CSS for background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F5DC; /* beige */
        color: #000000; /* white text color */
    }
    .css-1aumxhk {
        background-color: #000000 !important; /* white background for header */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the UI
st.title('Brain Tumor Classification')
st.write("""
    **Upload an MRI image of a brain, and the model will classify it into one of the following categories:**
    - Glioma Tumor
    - Meningioma Tumor
    - No Tumor
    - Pituitary Tumor
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display a spinner while processing the image
    with st.spinner('Processing...'):
        # Preprocess the image
        image = preprocess_image(uploaded_file)
        
        # Make prediction
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)
        
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Display the image name
        image_name = uploaded_file.name
        st.write(f"**Image Name:** {image_name}")
        
        # Display prediction and confidence
        st.write(f"**Predicted Class:** {tumor_classes[predicted_class]}")
        st.write(f"**Confidence:** {prediction[0][predicted_class]:.4f}")

        # Display confidence for all classes
        st.write("### Prediction Confidence for All Classes:")
        confidence_data = {
            "Class": tumor_classes,
            "Confidence": [f"{conf:.4f}" for conf in prediction[0]]
        }
        st.table(confidence_data)
        
        st.success('Classification completed!')

else:
    st.write("Please upload an MRI image file to get a classification.")

# Define the outro for additional context and user engagement
with st.expander("Read More"):
    st.markdown("""
        ## About This App
        This brain tumor classification application leverages deep learning techniques to provide insights into MRI images of the brain. 
        It aims to demonstrate the capabilities of machine learning in the field of medical imaging. The model used here has been trained on a dataset of MRI images 
        and is designed to classify images into four categories: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor.

        ## Usage Instructions
        - **Upload an Image**: Use the file uploader to select an MRI image in JPG, JPEG, or PNG format.
        - **View Predictions**: Once the image is uploaded, the app will process it and display the predicted class along with the confidence levels for each category.
        - **Educational Purpose**: Please note that this tool is intended for educational and demonstration purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

        ## Contact
        If you have any questions, suggestions, or feedback, feel free to reach out to the project maintainers at [yourname@gmail.com].

        Thank you for using our Brain Tumor Classification App!
    """)