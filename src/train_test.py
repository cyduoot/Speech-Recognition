from get_data import get_test_data
import tensorflow as tf
import os
import keras.backend.tensorflow_backend as KTF
from resnet import ResnetBuilder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K, Sequential


def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


a, b, c, d = get_test_data()
(X_train, y_train), (X_test, y_test) = (a, c), (b, d)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

KTF.set_session(get_session(0.8))
import numpy as np
np.random.seed(1337)  # for reproducibility

batch_size = 16
nb_classes = 20

nb_epoch = 12
# input image dimensions
img_rows, img_cols = 64 , 64
# number of convolutional filters to use
nb_filters = 16
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
r = np.random.permutation(len(y_train))
X_train = X_train[r, :] 
y_train = y_train[r]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# train a initial CNN
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
#model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
#model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
#model.add(BatchNormalization(axis=1))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=256, epochs=100,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
model.save("initial_2.h5")
print('Test score:', score[0])
print('Test accuracy:', score[1])

'''
#train a resnet
model=ResnetBuilder.build_resnet_18((1, 224, 224), 20)
model.summary()
model.compile(optimizer='sgd',loss='categorical_crossentropy')
model.fit(X_train, Y_train, batch_size=16, epochs=30,
          verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=2)
model.save("resnet_l.h5")
print(score)
try:
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
except:
	pass
'''