import numpy as np
import os
import pickle
import time

from keras import backend as K
from keras import regularizers
from keras.applications import vgg16
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Concatenate
from keras.layers import ConvLSTM2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import RepeatVector
from keras.layers import Conv2D
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from keras import Sequential
from utils import *

import cv2

context_model = vgg16.VGG16(input_shape=(1080, 1920, 3),
                                         include_top=False,
                                         weights='imagenet')   
img_data = load_img("E:\\PIEPredict-master\\PIE_dataset\\images\\set01\\video_0001\\01013.png")
image_array = img_to_array(img_data)
preprocessed_img = vgg16.preprocess_input(image_array)
expanded_img = np.expand_dims(preprocessed_img, axis=0)
img_features = context_model.predict(expanded_img)
y = Conv2D(filters=512, kernel_size=(3,3),strides=(5,9), activation='relu', input_shape=[33,60,512])(img_features)

print(y.shape)