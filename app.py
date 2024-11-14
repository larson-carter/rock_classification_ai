import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

# Function to download and load the model
@st.cache_resource
def download_model():
    # Google Drive direct download link
    model_url = 'https://drive.google.com/uc?id=1Vd4-_oWt7lKreYHhEa1ZlUd2VHQBnD-A'
    model_path = 'rock_classifier_model.h5'
    
    # Download the model if not present
    if not os.path.exists(model_path):
        try:
            st.write("Downloading the model. This may take a few moments...")
            gdown.download(model_url, model_path, quiet=False)
            st.write("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Error downloading the model: {e}")
            st.stop()
    
    # Load the model
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Load the model
model = download_model()

# App Title
st.title("Rock Classification AI")

# Define class labels (ensure this matches the training order)
class_labels = [
    'Wackestone', 'Blueschist', 'Evaporite', 'Granulite', 'Pyroxenite', 'Sandstone', 
    'Carbonatite', 'Shale', 'Granite', 'Serpentinite', 'Pumice', 'Pegmatite', 
    'Anthracite', 'Tuff', 'Migmatite', 'Quartz_monzonite', 'Rhyolite', 'Eclogite', 
    'Diamictite', 'Siltstone', 'Talc_carbonate', 'Greenschist', 'Tephrite', 
    'Quartzolite', 'Breccia', 'Slate', 'Limestone', 'Porphyry', 'Mudstone', 
    'Quartzite', 'Conglomerate', 'Flint', 'Gabbro', 'Basalt', 'Turbidite', 
    'Gneiss', 'Dolomite', 'Chalk', 'Scoria', 'Travertine', 'Chert', 'Marble', 
    'Coal', 'Amphibolite', 'Oolite', 'Phyllite', 'Andesite', 'Obsidian', 
    'Hornfels', 'Greywacke', 'Oil_shale', 'Komatiite', 'Quartz_diorite'
]

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        # Preprocess the image
        img_array = image.img_to_array(img) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class_index]
        predicted_class = class_labels[predicted_class_index]
        
        # Display prediction results
        st.write(f"**Predicted Rock Type:** {predicted_class}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        
        # Display additional information (ensure all classes have entries)
        rock_info_dict = {
            'Granite': 'Granite is a common type of felsic intrusive igneous rock, granular and phaneritic in texture.',
            'Basalt': 'Basalt is a mafic extrusive igneous rock formed from the rapid cooling of basaltic lava.',
            # Add descriptions for all other rock types...
            # Example:
            'Wackestone': 'Wackestone is a type of carbonate rock containing between 1% and 10% grains.',
            # Continue for all 52 classes...
        }
        
        # Fetch and display rock information
        rock_info = rock_info_dict.get(predicted_class, 'Information not available.')
        st.write(f"**Information:** {rock_info}")
    
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
