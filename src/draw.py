from get_data import get_test_data
import tensorflow as tf
import os
import keras.backend.tensorflow_backend as KTF
from resnet import ResnetBuilder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model
from keras import regularizers
import tensorflow as tf
from keras import layers
from keras.layers import Input,Add,Dense,Activation,ZeroPadding2D, \
    BatchNormalization,Flatten,Conv2D,AveragePooling2D,MaxPooling2D,GlobalMaxPooling2D
from resnet import ResnetBuilder
import keras.backend as K

model=ResnetBuilder.build_resnet_18((1, 224, 224), 20)
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=1)