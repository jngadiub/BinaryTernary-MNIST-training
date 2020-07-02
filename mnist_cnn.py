'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils

from mnist_mlp import print_model_to_json
from callbacks import all_callbacks

batch_size = 128
num_classes = 10
epochs = 2
# BN
epsilon = 1e-6
momentum = 0.9

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = np_utils.to_categorical(y_train, num_classes) * 2 - 1 # -1 or 1 for hinge loss
y_test = np_utils.to_categorical(y_test, num_classes) * 2 - 1

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softplus'))
model.add(Dense(num_classes))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))

model.summary()
print_model_to_json(model,'output_mnist_cnn/KERAS_model.json')

model.compile(loss=keras.losses.squared_hinge,
optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])

callbacks=all_callbacks(stop_patience=1000,
                           lr_factor=0.5,
                           lr_patience=10,
                           lr_epsilon=0.000001,
                           lr_cooldown=2,
                           lr_minimum=0.0000001,
                           outputDir='output_mnist_cnn')

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
               validation_split = 0.25, callbacks = callbacks.callbacks)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

