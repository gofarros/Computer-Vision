#import library
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow_hub.keras_layer import KerasLayer 
from tensorflow.keras.models import load_model

#load model
def run():
    # set title
    st.title('Project Worker Safety')
    image = Image.open('worker.jpg')
    st.image(image, caption = 'Photo: Pinterest')

    # File Uploader (Image)
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

    model = load_model('model_tl.keras', custom_objects={'KerasLayer': KerasLayer})
    target_size=(224, 224)

    def import_and_predict(image_data, model):
        image = load_img(image_data, target_size=target_size)
        img_array = img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Normalize the image
        img_array = img_array / 255.0

        # Make prediction
        predictions = model.predict(img_array)

        # Get the class with the highest probability
        idx = np.where(predictions >= 0.5, 1, 0).item()
        types = ['Not Safety', 'Safety']
        result = f"Prediction: {types[idx]}"

        return result

    if file is None:
        st.text("Please upload an image file")
    else:
        result = import_and_predict(file, model)
        st.image(file)
        st.write(result)
        
if __name__ == "__main__":
    run()