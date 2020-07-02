import sys, os, json, six, cv2

import numpy as np
import itertools

import keras.backend as K
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model, Sequential
from keras.metrics import categorical_accuracy

from binary_layers import BinaryDense, Clip, DropoutNoScaleForBinary, DropoutNoScale, BinaryConv2D
from binary_ops import binary_tanh as binary_tanh_op

from ternary_layers import TernaryDense, DropoutNoScaleForTernary
from ternary_ops import ternarize

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.colors as colors
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import pandas as pd

from optparse import OptionParser

from constraints import ZeroSomeWeights

from mnist_mlp import parse_config,get_features
from models import relu1

matplotlib.rcParams.update({'font.size': 16})

def model_up_to(model, n):
 m = Sequential()
 for layer in model.layers[:n]:
   m.add(layer)
 return m

def plot_layer_output(model):

    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation function

    # Testing
    print(inp.shape)
    test = np.random.random(784)

    layer_outs = [func([test]) for func in functors]
    print(layer_outs)

def add_logo(ax, fig, zoom, position='upper left', offsetx=10, offsety=10, figunits=False):

  #resize image and save to new file
  img = cv2.imread('logo.jpg', cv2.IMREAD_UNCHANGED)
  im_w = int(img.shape[1] * zoom )
  im_h = int(img.shape[0] * zoom )
  dim = (im_w, im_h)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  cv2.imwrite( "logo_resized.jpg", resized );

  #read resized image
  im = image.imread('logo_resized.jpg')
  im_w = im.shape[1]
  im_h = im.shape[0]

  #get coordinates of corners in data units and compute an offset in pixel units depending on chosen position
  ax_xmin,ax_ymax = 0,0
  offsetX = 0
  offsetY = 0
  if position=='upper left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[1]
   offsetX,offsetY = offsetx,-im_h-offsety
  elif position=='out left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[1]
   offsetX,offsetY = offsetx,offsety
  elif position=='upper right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
   offsetX,offsetY = -im_w-offsetx,-im_h-offsety
  elif position=='out right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
   offsetX,offsetY = -im_w-offsetx,offsety
  elif position=='bottom left':
   ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[0]
   offsetX,offsetY=offsetx,offsety
  elif position=='bottom right':
   ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[0]
   offsetX,offsetY=-im_w-offsetx,offsety
         
  #transform axis limits in pixel units
  ax_xmin,ax_ymax = ax.transData.transform((ax_xmin,ax_ymax))
  #compute figure x,y of bottom left corner by adding offset to axis limits (pixel units)
  f_xmin,f_ymin = ax_xmin+offsetX,ax_ymax+offsetY
       
  #add image to the plot
  plt.figimage(im,f_xmin,f_ymin)
       
  #compute box x,y of bottom left corner (= figure x,y) in axis/figure units
  if figunits:
   b_xmin,b_ymin = fig.transFigure.inverted().transform((f_xmin,f_ymin))
   #print("figunits",b_xmin,b_ymin)
  else: b_xmin,b_ymin = ax.transAxes.inverted().transform((f_xmin,f_ymin))
  
  #compute figure width/height in axis/figure units
  if figunits: f_xmax,f_ymax = fig.transFigure.inverted().transform((f_xmin+im_w,f_ymin+im_h)) #transform to figure units the figure limits
  else: f_xmax,f_ymax = ax.transAxes.inverted().transform((f_xmin+im_w,f_ymin+im_h)) #transform to axis units the figure limits
  b_w = f_xmax-b_xmin
  b_h = f_ymax-b_ymin

  #set which units will be used for the box
  transformation = ax.transAxes
  if figunits: transformation=fig.transFigure
  
  rectangle = FancyBboxPatch((b_xmin,b_ymin),
                              b_w, b_h,
                  transform=transformation,
                  boxstyle='round,pad=0.004,rounding_size=0.01',
                  facecolor='w',
                  edgecolor='0.8',linewidth=0.8,clip_on=False)
  ax.add_patch(rectangle)
    

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
    
    fig,ax = plt.subplots()
    for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred'] = predict_test[:,i]
        
        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])
        
        auc1[label] = auc(fpr[label], tpr[label])
        
        plt.plot(tpr[label],fpr[label],label='%s (%.4f)'%(label,auc1[label]))

    plt.semilogy()

    plt.xlabel("True positive rate")
    plt.ylabel("False positive rate")
    plt.ylim(0.001,1)
    plt.xlim(0.9,1)
    plt.grid(color='0.8', linestyle='dotted')
    plt.legend(loc='upper left',fontsize=15)
    plt.grid(color='0.8', linestyle='dotted')
    fig.tight_layout()
    plt.figtext(0.925, 0.94,options.label, wrap=True, horizontalalignment='right')
    add_logo(ax, fig, 0.14, position='upper right')
    plt.savefig('%s/ROC%s.png'%(outputDir,suffix))
    plt.savefig('%s/ROC%s.pdf'%(outputDir,suffix))
    return predict_test

def accuracy_per_class(cm):
    
 accs = np.zeros(10)
 accs_tot = 0
 for j in range(cm.shape[0] ):
  TP = 0.
  TN = 0.
  for i in range( cm.shape[0] ):
   if i != j:
    for k in range( cm.shape[0] ):
     if k!=j:
      TN+=cm[k][i]
   else:TP+=cm[j][i]
  print("----> Accuracy:",j,(TP+TN)/cm.sum(),"%.1f"%((TP+TN)/cm.sum()*100.))
  accs[j] = (TP+TN)/cm.sum()
  accs_tot+=(TP+TN)
 print("Min acc:",np.amin(accs),"Max acc:",np.amax(accs))

def per_class_accuracy(y_preds,y_true,class_labels):
 return [np.mean([(y_true[pred_idx] == np.round(y_pred)) for pred_idx, y_pred in enumerate(y_preds) if y_true[pred_idx] == int(class_label)]) for class_label in class_labels]
                
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    acc = 0
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(10):
         #print("From norm cfm --> Class:",i,"Acc: %.4f"%cm[i][i])
         acc+=cm[i][i]
    #print("Accuracy=%.4f"%(acc/10.))
        
    if normalize: plt.imshow(cm, cmap=cmap, norm=colors.LogNorm(vmin=0.0001, vmax=1.0)) #I had to hack /Users/jngadiub/anaconda3/lib/python3.6/site-packages/matplotlib/colors.py
    else: plt.imshow(cm, interpolation='nearest', cmap=cmap)

    #plt.title(title)
    cbar = plt.colorbar()
    plt.clim(0,1)
    cbar.set_label(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        txt  = format(cm[i, j], fmt)
        if normalize and txt=='0.00': txt = ""
        if normalize and txt!='0.00': txt = txt.replace('0.','.')
        plt.text(j, i, txt, horizontalalignment="center", verticalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize='13')

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
                   'ternary_tanh': ternary_tanh,
                   'Clip': Clip})
model.summary()

nb_classes = 10
X_train, X_test, Y_train, Y_test  = get_features(yamlConfig,nb_classes)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Keras test score:', score[0])
print('Keras test accuracy:', score[1])

labels = ['0','1','2','3','4','5','6','7','8','9']

Y_predict = makeRoc(X_test, labels, Y_test, model, outdir)

if options.convert:
 outfile_name = model_file.replace('.h5','_truth_labels.dat')
 print("Writing",Y_test.shape[1],"predicted labels for",Y_test.shape[0],"events in outfile",outfile_name)
 outfile = open(outfile_name,'w')
 for e in range(Y_test.shape[0]):
  line=''
  for l in range(Y_test.shape[1]): line+=(str(Y_test[e][l])+' ')
  outfile.write(line+'\n')
 outfile.close()

 outfile_name = model_file.replace('.h5','_input_features.dat')
 print("Writing",X_test.shape[1],"input features for",X_test.shape[0],"events in outfile",outfile_name)
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
 print("Writing",Y_predict.shape[1],"predictions for",Y_predict.shape[0],"events in outfile",outfile_name)
 outfile = open(outfile_name,'w')
 for e in range(Y_predict.shape[0]):
  line=''
  for l in range(Y_predict.shape[1]): line+=(str(Y_predict[e][l])+' ')
  outfile.write(line+'\n')
 outfile.close()

y_test_proba = Y_test.argmax(axis=1)
y_predict_proba = Y_predict.argmax(axis=1)
cnf_matrix = confusion_matrix(y_test_proba, y_predict_proba)
print("ACCURACY PER CLASS")
accuracy_per_class(cnf_matrix)

np.set_printoptions(precision=2)

labels = ['0','1','2','3','4','5','6','7','8','9']

fig,ax = plt.subplots()
plot_confusion_matrix(cnf_matrix, classes=[l for l in labels],title='Confusion matrix')
fig.tight_layout()
plt.figtext(0.75, 0.9,options.label, wrap=True, horizontalalignment='right')
add_logo(ax, fig, 0.14, position='out left', offsetx=28, offsety=8, figunits=True)
plt.savefig(outdir+"/confusion_matrix.png")
plt.savefig(outdir+"/confusion_matrix.pdf")

fig,ax = plt.subplots()
plot_confusion_matrix(cnf_matrix, classes=[l for l in labels], normalize=True,title='Normalized confusion matrix')
plt.subplots_adjust(left=0, bottom=0.13)
plt.figtext(0.72, 0.9,options.label, wrap=True, horizontalalignment='right')
add_logo(ax, fig, 0.14, position='out left', offsetx=102, offsety=8, figunits=True)
plt.savefig(outdir+"/confusion_matrix_norm.png")
plt.savefig(outdir+"/confusion_matrix_norm.pdf")

if options.checkHLS != '':
    
    Y_hls = np.loadtxt(options.checkHLS)
    
    labels = ['zero','one','two','three','four','five','six','seven','eight','nine']
    makeRoc(X_test, labels, Y_test, model, outdir, Y_hls,'_hls')
    
    y_predict_proba = Y_hls.argmax(axis=1)
    print('HLS test accuracty (scikit-learn):',accuracy_score(y_test_proba, y_predict_proba))
    cnf_matrix = confusion_matrix(y_test_proba, y_predict_proba)
    np.set_printoptions(precision=2)

    accuracy_per_class(cnf_matrix)
    
    labels = ['0','1','2','3','4','5','6','7','8','9']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[l for l in labels],title='Confusion matrix')
    plt.figtext(0.28, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.75, 0.9,options.label, wrap=True, horizontalalignment='right', fontsize=12)
    plt.savefig(outdir+"/confusion_matrix_hls.png")

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[l for l in labels], normalize=True,title='Normalized confusion matrix')
    plt.figtext(0.28, 0.90,'hls4ml',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.75, 0.9,options.label, wrap=True, horizontalalignment='right', fontsize=12)
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
