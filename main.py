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

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import collections.abc
import functools
from typing import Any, Callable, Iterable, Iterator, Union

import numpy as np

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

from util import classify, set_background



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

model_path = r'fine_tuned_xray_model_280.keras'

# Load the model
model = tf.keras.models.load_model(model_path)
#model = load_model('./model/pneumonia_classifier.h5')

# load class names
#with open('./model/labels.txt', 'r') as f:
    #class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    #f.close()
class_name=''
conf_score=0
class_names = ['Normal', 'Unhealthy']
# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score, percentage = classify(image, model, class_names)

    # write classification
    #st.write("## {}".format(class_name))
    #st.write("### score: {}%".format(int(conf_score * 1000) / 10))
import streamlit as st
# Use st.markdown to set text color to black
st.markdown(f"## <span style='color: black;'>{class_name}</span>", unsafe_allow_html=True)
st.markdown(f"### <span style='color: black;'>score: {int(conf_score * 1000) / 10}%</span>", unsafe_allow_html=True)
###########################
st.markdown(f"### <span style='color: black;'>score: {int(percentage)}%</span>", unsafe_allow_html=True)
# Print probabilities as percentages

