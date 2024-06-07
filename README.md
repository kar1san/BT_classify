
The Brain Tumor Classification App is a web application built using Streamlit and TensorFlow. It allows users to upload MRI images of the brain, which are then classified into one of four categories: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor. The app leverages a pre-trained deep learning model to perform the classification.



Features
Image Upload: Users can upload MRI images in JPG, JPEG, or PNG format.
Image Preprocessing: Uploaded images are resized and normalized before being fed into the model.
Prediction: The app displays the predicted class along with the confidence level.
Visualization: The uploaded image is displayed, and prediction confidence for all classes is shown in a table format.
Interactive UI: The app features an interactive UI with custom styling and an expandable section for additional information.


Installation
Prerequisites
Python 3.x
Streamlit
TensorFlow
Pillow
Numpy
