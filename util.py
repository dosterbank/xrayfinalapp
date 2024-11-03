import base64
#import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow import keras
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import plotly.express as px
import scipy as sp
import tensorflow as tf
from scipy import ndimage
from shutil import copyfile
from tensorflow.keras.layers import Conv2D,Add,MaxPooling2D, Dense, BatchNormalization,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import plotly.express as px
from tensorflow.keras.applications import VGG19
#predictions_str = ""
last_conv_layer_name = None
image=None
###############################################################################
def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

##############################################################################################
def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    predictions_str = ""
    image_index=0
    # Resize image to (280, 280)
    image = ImageOps.fit(image, (280, 280), Image.Resampling.LANCZOS)
    #img = load_img(image_path, target_size=(280, 280))  # Adjust size if different
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Scale pixel values to [0, 1] if your model requires it
    # Make a prediction
    results = model.predict(img_array)
    #rint(f"Selected image: {results}")rescale=1./255)
    #gradcam_and_plot(model, img_array, results, image_index, class_names)
    prediction = np.argmax(results[image_index]) # returns the class predictes as number
    #print(f"prediction: {prediction}")
    for class_name, probability in zip(class_names, results[0]):
        percentage = probability * 100  # Convert to percentage
        predictions_str += f"{class_name}: {percentage:.2f}%\n"  # Format to two decimal places and add newline
    # Now you can display `predictions_str` using Streamlit
    return  predictions_str, img_array

################################

#print("Last convolutional layer name:", last_conv_layer_name)
############################################################

##########################################################################