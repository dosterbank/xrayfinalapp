import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pathlib

import os
from PIL import Image
import tensorflow as tf
import keras
import keras.callbacks
from keras.callbacks import TensorBoard
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from keras.models import Sequential
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, BatchNormalization, SimpleRNN, Flatten, LSTM
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter
from numpy import where
from matplotlib import pyplot
import matplotlib.pyplot as plt

import torch,time,os, shutil
#import models
import utils
import pandas as pd
import numpy as np
import torch, time, os, shutil
#import models, utils
import numpy as np
import pandas as pd
import torch
#from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader
import tensorflow as tf
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# Setup feedback system
#from learntools.core import binder
#binder.bind(globals())
#from learntools.deep_learning_intro.ex3 import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
# Setup feedback system



#from learntools.core import binder
#binder.bind(globals())
#from learntools.computer_vision.ex5 import *

# Imports
import keras
import keras.callbacks
from keras.callbacks import TensorBoard
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import collections.abc
import functools
from typing import Any, Callable, Iterable, Iterator, Union

import numpy as np
import matplotlib as mpl
import tensorflow as tf
from tensorflow_datasets.core import tf_compat
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.utils import type_utils
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report
import streamlit as st
from util import classify, set_background
from tensorflow.keras.preprocessing.image import load_img, img_to_array
predictions_str=""
class_name=''
image_index=None
image_file=None
#image_index = 0  # Since you are using a single image, the index is 0
#####################################
set_background('./bgs/yachaylogo.png')
##########################################
# Title
#st.title("VGG19 Chest X ray  Diagnosis Tool")

# Red text using HTML
st.markdown('<h1 style="color: black;">VGG19 Chest X ray  Diagnosis Tool</h1>', unsafe_allow_html=True)

# Blue text using HTML
st.markdown('<p style="color: blue;">Please upload a chest X-ray image.</p>', unsafe_allow_html=True)

# Using custom CSS
st.markdown(
    """
    <style>
    .green-text {
        color: green;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#st.markdown('<p class="green-text">URL.....</p>', unsafe_allow_html=True)

##########################################
# set title
#st.title('VGG19 Chest X ray  Diagnosis Tool')

# set header
#st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])
# Display the filename in black if a file is uploaded
if file is not None:
    st.markdown(f"<p style='color: black;'>Opened file: <strong>{file.name}</strong></p>", unsafe_allow_html=True)
    # Optionally display the uploaded file
    #st.image(file, caption='Uploaded Image', use_column_width=True)
# load classifier
#####################################################################
model_path = r'fine_tuned_xray_model_280.keras'

# Load the model
model = tf.keras.models.load_model(model_path)
for layer in reversed(model.layers):
    if 'conv' in layer.name:  # Busca capas con 'conv' en su nombre
        last_conv_layer_name = layer.name
        break
# Create a model that maps the input image to the activations of the last conv layer and the output predictions
grad_model = tf.keras.models.Model(
    model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
)

#model = load_model('./model/pneumonia_classifier.h5')

# load class names
#with open('./model/labels.txt', 'r') as f:
    #class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    #f.close()

#conf_score=0
class_names = ['Normal', 'Unhealthy']
# $$$#########################################333333333####################
#                       display image

if file is not None:
    image = Image.open(file).convert('RGB') ########## image is the var
    st.image(image, use_column_width=True)  # muestra en pantalla image

    # classify image funcion 
    #class_name, conf_score, percentage = classify(image, model, class_names) 
    predictions_str, img_array=classify(image, model, class_names)


############################### Print results ############################
# Load and preprocess the image
#img = load_img(img_array, target_size=(280, 280))  # Adjust size if different
#img_array = img_to_array(img)
#img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#img_array /= 255.0  # Scale pixel values to [0, 1] if your model requires it

last_conv_layer_name = None

for layer in reversed(model.layers):
    if 'conv' in layer.name:  # Busca capas con 'conv' en su nombre
        last_conv_layer_name = layer.name
        break

print("Last convolutional layer name:", last_conv_layer_name)
# Make a prediction
results = model.predict(img_array)
print(f"Selected image: {results}")
#gradcam_and_plot(model, img_array, results, image_index, class_names)
prediction = np.argmax(results[image_index])
print(f"prediction: {prediction}")
############################################cehc aca############################
cam_path="/cam.jpg" 
alpha=0.4
##################################################
# Compute the gradient of the top predicted class for the input image
with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)

    # Debugging information
    print("Last Conv Layer Output Shape:", last_conv_layer_output.shape)
    print("Predictions Shape:", preds.shape)

    # Check if preds is empty
    if tf.size(preds) == 0:
        raise ValueError("Model produced no predictions.")
    
    # Get the index of the predicted class
    pred_index = tf.argmax(preds[0]).numpy()  # Ensure pred_index is an integer
    print("Predicted Index:", pred_index)

    # Access the class channel corresponding to the predicted index
    class_channel = preds[:, pred_index]

# Now you can proceed with computing gradients and creating the heatmap
grads = tape.gradient(class_channel, last_conv_layer_output)

# This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# Multiply each channel in the feature map array by "how important this channel is"
# then sum all the channels to obtain the heatmap class activation
last_conv_layer_output = last_conv_layer_output[0]
heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize the heatmap between 0 & 1 for visualization
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

# Print the heatmap shape for confirmation
print("Heatmap Shape:", heatmap.shape)
heatmap.numpy()
# Display heatmap
plt.matshow(heatmap)
plt.savefig("heatmap.png")
######################################## continue
#img = keras.utils.load_img(image)
img = keras.utils.img_to_array(image)

    # Rescale heatmap to a range 0-255
heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
jet_heatmap = keras.utils.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
superimposed_img = jet_heatmap * alpha + img
superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
#superimposed_img.save(cam_path)

    # Display Grad CAM
    #display(Image(cam_path))

############################################# Print all the probabilities as percentages
#st.text(predictions_str)
#st.markdown(f"## <span style='color: black;'>{predictions_str}</span>", unsafe_allow_html=True)
# Assuming `predictions_str` already has the formatted text for all classes
predictions_str = predictions_str.replace("\n", "<br>")  # Replace newline with HTML line break

# Display predictions with Streamlit, ensuring both classes are shown with their accuracies
st.markdown(
    f"## <span style='color: black;'>{predictions_str}</span>", 
    unsafe_allow_html=True
)
st.image(superimposed_img, use_column_width=True)  # muestra en pantall

##################

