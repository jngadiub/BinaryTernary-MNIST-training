import sys, os, json, six

import numpy as np
import itertools

import keras.backend as K
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model

from binary_layers import BinaryDense, Clip, DropoutNoScaleForBinary, DropoutNoScale, BinaryConv2D
from binary_ops import binary_tanh as binary_tanh_op

from ternary_layers import TernaryDense, DropoutNoScaleForTernary
from ternary_ops import ternarize

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

from optparse import OptionParser

from constraints import ZeroSomeWeights
from sklearn.metrics import accuracy_score

from mnist_mlp import parse_config,get_features
from models import relu1

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

def makeRoc(features_val, labels, labels_val, model, outputDir, predict_hls = [], suffix=''):

    if len(predict_hls) == 0: predict_test = model.predict(features_val)
    else: predict_test = predict_hls
    
    df = pd.DataFrame()
    
    fpr = {}
    tpr = {}
    auc1 = {}
    
    plt.figure()
    for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred'] = predict_test[:,i]
        
        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])
        
        auc1[label] = auc(fpr[label], tpr[label])
        
        #plt.plot(tpr[label],fpr[label],label='%s, AUC = %.1f%%'%(label,auc1[label]*100.))
        plt.plot(fpr[label],tpr[label],label='%s, AUC = %.1f%%'%(label,auc1[label]*100.))

    #plt.semilogy()
    #plt.xlabel("Signal Efficiency")
    #plt.ylabel("Background Efficiency")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.ylim(0.9,1.01)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.figtext(0.24, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    #plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.figtext(0.9, 0.9,options.label, wrap=True, horizontalalignment='right', fontsize=12)
    plt.savefig('%s/ROC%s.png'%(outputDir,suffix))
    return predict_test

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

def binary_tanh(x):
    return binary_tanh_op(x)

def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)

parser = OptionParser()
parser.add_option('-c','--config'   ,action='store',type='string',dest='config'   ,default='mnist.yml', help='yaml config file')
parser.add_option('-l','--label',action='store',type='string',dest='label',default='Binary MNIST-128',help='label for plots')
parser.add_option('-C','--convert',action='store_true',dest='convert',default=False, help='convert data to txt')
parser.add_option('--checkHLS','--checkHLS',action='store',type='string',dest='checkHLS',default='', help='file with HLS output')
(options,args) = parser.parse_args()

yamlConfig = parse_config(options.config)

outdir = yamlConfig['OutputDir']
model_file = outdir+'/KERAS_check_best_model.h5'
model = load_model(model_file, custom_objects={
                   'relu1': relu1,
                   'ZeroSomeWeights' : ZeroSomeWeights,
                   'DropoutNoScale':DropoutNoScale,
                   'DropoutNoScaleForBinary':DropoutNoScaleForBinary,
                   'DropoutNoScaleForTernary':DropoutNoScaleForTernary,
                   'Clip': Clip,
                   'BinaryDense': BinaryDense,
                   'TernaryDense': TernaryDense,
		   'BinaryConv2D': BinaryConv2D,
                   'binary_tanh': binary_tanh,
                   'ternary_tanh': ternary_tanh})

nb_classes = 10
X_train, X_test, Y_train, Y_test  = get_features(yamlConfig,nb_classes)

score = model.evaluate(X_test, Y_test, verbose=0)
print 'Keras test score:', score[0]
print 'Keras test accuracy:', score[1]

labels = ['zero','one','two','three','four','five','six','seven','eight','nine']

Y_predict = makeRoc(X_test, labels, Y_test, model, outdir)

if options.convert:
 outfile_name = model_file.replace('.h5','_truth_labels.dat')
 print "Writing",Y_test.shape[1],"predicted labels for",Y_test.shape[0],"events in outfile",outfile_name
 outfile = open(outfile_name,'w')
 for e in range(Y_test.shape[0]):
  line=''
  for l in range(Y_test.shape[1]): line+=(str(Y_test[e][l])+' ')
  outfile.write(line+'\n')
 outfile.close()

 outfile_name = model_file.replace('.h5','_input_features.dat')
 if not 'cnn' in yamlConfig['KerasModel']: print "Writing",X_test.shape[1],"input features for",X_test.shape[0],"events in outfile",outfile_name
 else: print "Writing",X_test.shape[1]*X_test.shape[2]*X_test.shape[3],"input features for",X_test.shape[0],"events in outfile",outfile_name
 outfile = open(outfile_name,'w')
 for e in range(X_test.shape[0]):
  line=''
  for w in range(X_test.shape[1]):
   if 'cnn' in yamlConfig['KerasModel']:
    for h in range(X_test.shape[2]):
     for c in range(X_test.shape[3]): line+=(str(X_test[e][w][h][c])+' ')
   else: line+=(str(X_test[e][w])+' ')
  outfile.write(line+'\n')
 outfile.close()

 outfile_name = model_file.replace('.h5','_predictions.dat')
 print "Writing",Y_predict.shape[1],"predictions for",Y_predict.shape[0],"events in outfile",outfile_name
 outfile = open(outfile_name,'w')
 for e in range(Y_predict.shape[0]):
  line=''
  for l in range(Y_predict.shape[1]): line+=(str(Y_predict[e][l])+' ')
  outfile.write(line+'\n')
 outfile.close()

y_test_proba = Y_test.argmax(axis=1)
y_predict_proba = Y_predict.argmax(axis=1)
cnf_matrix = confusion_matrix(y_test_proba, y_predict_proba)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[l for l in labels],title='Confusion matrix')
plt.figtext(0.28, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
plt.figtext(0.75, 0.9,options.label, wrap=True, horizontalalignment='right', fontsize=12)
#plt.figtext(0.38, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
plt.savefig(outdir+"/confusion_matrix.png")

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[l for l in labels], normalize=True,title='Normalized confusion matrix')
plt.figtext(0.28, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
plt.figtext(0.75, 0.9,options.label, wrap=True, horizontalalignment='right', fontsize=12)
#plt.figtext(0.38, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
plt.savefig(outdir+"/confusion_matrix_norm.png")

if options.checkHLS != '':
    
    Y_hls = np.loadtxt(options.checkHLS)
    score = model.evaluate(X_test, Y_hls, verbose=0)
    print 'HLS test score:', score[0]
    print 'HLS test accuracy:', score[1]
    #print 'HLS test accuracty (scikit-learn):',accuracy_score(Y_test, Y_hls)
    
    makeRoc(X_test, labels, Y_test, model, outdir,Y_hls,'_hls')
    
    y_test_proba = Y_hls.argmax(axis=1)
    cnf_matrix = confusion_matrix(y_test_proba, y_predict_proba)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[l for l in labels],title='Confusion matrix')
    plt.figtext(0.28, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.75, 0.9,options.label, wrap=True, horizontalalignment='right', fontsize=12)
    #plt.figtext(0.38, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(outdir+"/confusion_matrix_hls.png")

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[l for l in labels], normalize=True,title='Normalized confusion matrix')
    plt.figtext(0.28, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.75, 0.9,options.label, wrap=True, horizontalalignment='right', fontsize=12)
    #plt.figtext(0.38, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14)
    plt.savefig(outdir+"/confusion_matrix_norm_hls.png")

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
