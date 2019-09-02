'''Trains a simple binarize fully connected NN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 97.9% test accuracy after 20 epochs using theano backend
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys, os, yaml
from optparse import OptionParser

import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

from callbacks import all_callbacks
from constraints import *

import h5py
import models

def get_features(yamlConfig,nclasses):
    
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

    if yamlConfig['KerasLoss'] == 'squared_hinge':
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nclasses) * 2 - 1 # -1 or 1 for hinge loss
        Y_test = np_utils.to_categorical(y_test, nclasses) * 2 - 1
    else:
        Y_train = np_utils.to_categorical(y_train, nclasses)
        Y_test = np_utils.to_categorical(y_test, nclasses)

    return X_train,X_test,Y_train,Y_test

def print_model_to_json(keras_model, outfile_name):
    outfile = open(outfile_name,'w')
    jsonString = keras_model.to_json()
    import json
    with outfile:
        obj = json.loads(jsonString)
        json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
        outfile.write('\n')

def parse_config(config_file) :
    
    print("Loading configuration from", config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

if __name__ == "__main__":
 parser = OptionParser()
 parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='mnist.yml', help='yaml config file')
 (options,args) = parser.parse_args()

 yamlConfig = parse_config(options.config)
 
 outdir = yamlConfig['OutputDir']
 if not os.path.exists(outdir): os.system('mkdir '+outdir)
 else: raw_input("Warning: output directory exists. Press Enter to continue...")
 
 print("OUTDIR:",outdir)
 print("MODEL:",yamlConfig['KerasModel'])
 print("NEURONS:",yamlConfig['Neurons'])
 print("LAYERS:",yamlConfig['Layers'])
 print("LOSS:",yamlConfig['KerasLoss'])
 print("L1 REGULARIZATION",yamlConfig['L1Reg'])
 print("EPOCHS:",yamlConfig['Epochs'])

 batch_size = 100
 epochs = yamlConfig['Epochs']
 nb_classes = 10

 #H = 'Glorot'
 #kernel_lr_multiplier = 'Glorot'

 H = 1.0
 kernel_lr_multiplier = 1.0

 # network
 num_unit = 128
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

 X_train, X_test, Y_train, Y_test  = get_features(yamlConfig,nb_classes)

 try:
  print("RETRAIN:",yamlConfig['KerasModelRetrain'])
  model = getattr(models, yamlConfig['KerasModelRetrain'])
  keras_model = model(Input(shape=X_train.shape[1:]),num_unit,num_hidden,nb_classes,drop_in,drop_hidden,epsilon,momentum,l1Reg=yamlConfig['L1Reg'],h5fName=yamlConfig['Constraints'])
  keras_mode.load_weights(yamlConfig['Constraints'].replace('_drop_weights.h5','.h5'), by_name=True)
 except:
  model = getattr(models, yamlConfig['KerasModel'])
  keras_model = model(Input(shape=X_train.shape[1:]),num_unit,num_hidden,nb_classes,drop_in,drop_hidden,epsilon,momentum,l1Reg=yamlConfig['L1Reg'])

 keras_model.summary()
 print_model_to_json(keras_model,outdir + '/KERAS_model.json')

 opt = Adam(lr=lr_start)
 keras_model.compile(loss=yamlConfig['KerasLoss'], optimizer=opt, metrics=['acc'])

 callbacks=all_callbacks(stop_patience=1000,
                            lr_factor=0.5,
                            lr_patience=10,
                            lr_epsilon=0.000001,
                            lr_cooldown=2,
                            lr_minimum=0.0000001,
                            outputDir=outdir)

 keras_model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
                validation_split = 0.25, callbacks = callbacks.callbacks)

 score = keras_model.evaluate(X_test, Y_test, verbose=0)
 print('Test score:', score[0])
 print('Test accuracy:', score[1])
