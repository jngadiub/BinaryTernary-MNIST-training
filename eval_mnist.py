import sys, os, json, six

import numpy as np
import itertools

import keras.backend as K
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model

from binary_layers import BinaryDense, Clip
from binary_ops import binary_tanh as binary_tanh_op

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, six.text_type):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.items()
    }
    # if it's anything else, return it in its original form
    return data

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0,1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class DropoutNoScale(Dropout):
    '''Keras Dropout does scale the input in training phase, which is undesirable here.
        '''
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)
            
            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

def binary_tanh(x):
    return binary_tanh_op(x)

outdir = str(sys.argv[1])
model_file = outdir+'/KERAS_check_best_model.h5'
model = load_model(model_file, custom_objects={
                   'DropoutNoScale':DropoutNoScale,
                       'BinaryDense': BinaryDense,
                       'binary_tanh': binary_tanh,
                       'Clip': Clip})

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print X_train.shape[0], 'train samples'
print X_test.shape[0], 'test samples'

nb_classes = 10

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1

score = model.evaluate(X_test, Y_test, verbose=0)
print 'Test score:', score[0]
print 'Test accuracy:', score[1]

outfile_name = model_file.replace('.h5','_truth_labels.dat')
print "Writing",Y_test.shape[1],"predicted labels for",Y_test.shape[0],"events in outfile",outfile_name
outfile = open(outfile_name,'w')
for e in range(Y_test.shape[0]):
 line=''
 for l in range(Y_test.shape[1]): line+=(str(Y_test[e][l])+' ')
 outfile.write(line+'\n')
outfile.close()

outfile_name = model_file.replace('.h5','_input_features.dat')
print "Writing",X_test.shape[1],"input features for",X_test.shape[0],"events in outfile",outfile_name
outfile = open(outfile_name,'w')
for e in range(X_test.shape[0]):
 line=''
 for l in range(X_test.shape[1]): line+=(str(X_test[e][l])+' ')
 outfile.write(line+'\n')
outfile.close()

Y_predict = model.predict(X_test)
outfile_name = model_file.replace('.h5','_predictions.dat')
print "Writing",Y_predict.shape[1],"predictions for",Y_predict.shape[0],"events in outfile",outfile_name
outfile = open(outfile_name,'w')
for e in range(Y_predict.shape[0]):
 line=''
 for l in range(Y_predict.shape[1]): line+=(str(Y_predict[e][l])+' ')
 outfile.write(line+'\n')
outfile.close()

labels = ['zero','one','two','three','four','five','six','seven','eight','nine']
y_test_proba = Y_test.argmax(axis=1)
y_predict_proba = Y_predict.argmax(axis=1)
cnf_matrix = confusion_matrix(y_test_proba, y_predict_proba)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[l for l in labels],title='Confusion matrix')
plt.figtext(0.28, 0.90,'MNIST',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
#plt.figtext(0.38, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
plt.savefig(outdir+"/confusion_matrix.png")
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[l for l in labels], normalize=True,title='Normalized confusion matrix')
                          
plt.figtext(0.28, 0.90,'MNIST',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
#plt.figtext(0.38, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
plt.savefig(outdir+"/confusion_matrix_norm.png")

if os.path.isfile(outdir+'/full_info.log'):
        f = open(outdir+'/full_info.log')
        myListOfDicts = json.load(f, object_hook=_byteify)
        myDictOfLists = {}
        for key, val in myListOfDicts[0].items():
            myDictOfLists[key] = []
        for i, myDict in enumerate(myListOfDicts):
            for key, val in myDict.items():
                myDictOfLists[key].append(myDict[key])
    
        plt.figure()
        val_loss = np.asarray(myDictOfLists[b'val_loss'])
        loss = np.asarray(myDictOfLists[b'loss'])
        plt.plot(val_loss, label='validation')
        plt.plot(loss, label='train')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(outdir+"/loss.png")
        
        plt.figure()
        val_acc = np.asarray(myDictOfLists[b'val_acc'])
        acc = np.asarray(myDictOfLists[b'acc'])
        plt.plot(val_acc, label='validation')
        plt.plot(acc, label='train')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig(outdir+"/acc.png")
