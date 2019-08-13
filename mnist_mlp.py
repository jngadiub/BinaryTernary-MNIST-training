'''Trains a simple binarize fully connected NN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 97.9% test accuracy after 20 epochs using theano backend
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys, os
from optparse import OptionParser

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from callbacks import all_callbacks

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, Clip, DropoutNoScaleForBinary
from ternary_ops import ternarize
from ternary_layers import TernaryDense, DropoutNoScaleForTernary

from keras.models import load_model

def print_model_to_json(keras_model, outfile_name):
    outfile = open(outfile_name,'w')
    jsonString = keras_model.to_json()
    import json
    with outfile:
        obj = json.loads(jsonString)
        json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        outfile.write('\n')

def binary_tanh(x):
    return binary_tanh_op(x)

def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)

parser = OptionParser()
parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='train_mnist/', help='output directory')
parser.add_option('-p','--precision'   ,action='store',type='string',dest='precision'   ,default='binary', help='binary, ternary, or float')
parser.add_option('--relu','--relu'   ,action='store_true',dest='relu'   ,default=False, help='use relu')
(options,args) = parser.parse_args()

outdir = options.outputDir
use_relu = options.relu
if not os.path.exists(outdir): os.system('mkdir '+outdir)
else: print("Directory",outdir,'already exists!')

print("OUTDIR:",outdir)
print("USE RELU:",use_relu)
print("PRECISION:",options.precision)

batch_size = 100
epochs = 100
nb_classes = 10

#H = 'Glorot'
#kernel_lr_multiplier = 'Glorot'

H = 1.0
kernel_lr_multiplier = 1.0

# network
num_unit = 512
num_hidden = 3
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
drop_in = 0.2
drop_hidden = 0.5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1

model = Sequential()
if options.precision == 'binary':
 model.add(DropoutNoScaleForBinary(drop_in, input_shape=(784,), name='drop0'))
 for i in range(num_hidden):
    model.add(BinaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,name='dense{}'.format(i+1)))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
    if use_relu: model.add(Activation('relu', name='act{}'.format(i+1)))
    else: model.add(Activation(binary_tanh, name='act{}'.format(i+1)))
    model.add(DropoutNoScaleForBinary(drop_hidden, name='drop{}'.format(i+1)))
 model.add(BinaryDense(10, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,name='dense'))
 model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))

if options.precision == 'ternary':
 model.add(DropoutNoScaleForTernary(drop_in, input_shape=(784,), name='drop0'))
 for i in range(num_hidden):
    model.add(TernaryDense(num_unit, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,name='dense{}'.format(i+1)))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
    if use_relu: model.add(Activation('relu', name='act{}'.format(i+1)))
    else: model.add(Activation(ternary_tanh, name='act{}'.format(i+1)))
    model.add(DropoutNoScaleForTernary(drop_hidden, name='drop{}'.format(i+1)))
 model.add(TernaryDense(10, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias,name='dense'))
 model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))

if options.precision == 'float':
 model.add(Dropout(drop_in, input_shape=(784,), name='drop0'))
 for i in range(num_hidden):
    model.add(Dense(num_unit,name='dense{}'.format(i+1)))
    model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1)))
    model.add(Activation('relu', name='act{}'.format(i+1)))
    model.add(Dropout(drop_hidden, name='drop{}'.format(i+1)))
 model.add(Dense(10, activation='softmax',name='dense'))
 model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))

model.summary()
print_model_to_json(model,outdir + '/KERAS_model.json')

opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])

# deserialized custom layers
#model.save('mlp.h5')
#model = load_model('mlp.h5', custom_objects={'DropoutNoScale': DropoutNoScale,
#                                             'BinaryDense': BinaryDense,
#                                             'Clip': Clip, 
#                                             'binary_tanh': binary_tanh})

#lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
callbacks=all_callbacks(stop_patience=1000,
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001,
                            lr_cooldown=2,
                            lr_minimum=0.0000001,
                            outputDir=outdir)
#history = model.fit(X_train, Y_train,
#                batch_size=batch_size, epochs=epochs,
#                verbose=1, validation_data=(X_test, Y_test),
#                callbacks=[lr_scheduler])
model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
                validation_split = 0.25, callbacks = callbacks.callbacks)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
