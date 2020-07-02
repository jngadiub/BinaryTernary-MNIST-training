from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, Clip, DropoutNoScaleForBinary, BinaryConv2D
from ternary_ops import ternarize
from ternary_layers import TernaryDense, DropoutNoScaleForTernary
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, Conv2D, MaxPooling2D, Flatten
from keras.regularizers import l1
from keras.models import Model
import keras.backend as K

def binary_tanh(x):
    return binary_tanh_op(x)

def ternary_tanh(x):
    x = K.clip(x, -1, 1)
    return ternarize(x)

def relu1(x):
    return K.relu(x, max_value=1.)

def float_model_softmax(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
 
 print("CREATING FLOAT MNIST MODEL WITH",num_unit,"NEURONS AND",num_hidden,"LAYERS")
 
 if l1Reg!=0: x = Dropout(drop_in, name='drop0')(Inputs)
 else: x = Dense(num_unit,name='dense1',W_regularizer=l1(l1Reg))(Inputs)
 for i in range(1,num_hidden):
   x = Dense(num_unit,activation='relu',name='dense{}'.format(i+1),W_regularizer=l1(l1Reg))(x)
   if l1Reg!=0: x = Dropout(drop_hidden, name='drop{}'.format(i+1))(x)
 predictions = Dense(nclasses, activation='softmax',name='dense')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def float_model_softmax_constraint(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0,h5fName=None):
    
  print("RETRAIN FLOAT MNIST MODEL WITH DROPPED WEIGHTS")
  h5f = h5py.File(h5fName)
  
  x = Dense(num_unit,name='dense0',W_regularizer=l1(l1Reg),kernel_constraint=zero_some_weights(binary_tensor=h5f['dense0'][()].tolist()))(Inputs)
  for i in range(num_hidden):
    x = Dense(num_unit,activation='relu',name='dense{}'.format(i+1),W_regularizer=l1(l1Reg),kernel_constraint=zero_some_weights(binary_tensor=h5f['dense{}'.format(i+1)][()].tolist()))(x)
  predictions = Dense(nclasses, activation='softmax',name='dense',kernel_constraint=zero_some_weights(binary_tensor=h5f['dense'][()].tolist()))(x)
  model = Model(inputs=Inputs, outputs=predictions)
  return model

def float_model_softmax_batchnorm(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING FLOAT MNIST MODEL WITH",num_unit,"NEURONS AND",num_hidden,"LAYERS")

 if l1Reg!=0: x = Dropout(drop_in, name='drop0')(Inputs)
 else: x = Dense(num_unit,name='dense0',W_regularizer=l1(l1Reg))(Inputs)
 for i in range(num_hidden):
   x = Dense(num_unit,name='dense{}'.format(i+1),W_regularizer=l1(l1Reg))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation('relu', name='act{}'.format(i+1))(x)
   if l1Reg!=0: x = Dropout(drop_hidden, name='drop{}'.format(i+1))(x)
 predictions = Dense(nclasses, activation='softmax',name='dense')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def float_model_softmax_batchnorm_constraint(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0,h5fName=None):
    
 print("RETRAIN FLOAT MNIST MODEL WITH DROPPED WEIGHTS")
 h5f = h5py.File(h5fName)
    
 x = Dense(num_unit,name='dense0',W_regularizer=l1(l1Reg),kernel_constraint=zero_some_weights(binary_tensor=h5f['dense0'][()].tolist()))(Inputs)
 for i in range(num_hidden):
   x = Dense(num_unit,name='dense{}'.format(i+1),W_regularizer=l1(l1Reg),kernel_constraint=zero_some_weights(binary_tensor=h5f['dense{}'.format(i+1)][()].tolist()))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation('relu', name='act{}'.format(i+1))(x)
 predictions = Dense(nclasses, activation='softmax',name='dense',kernel_constraint=zero_some_weights(binary_tensor=h5f['dense'][()].tolist()))(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def float_model_hinge(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING FLOAT MNIST MODEL WITH",num_unit,"NEURONS AND",num_hidden,"LAYERS")

 if l1Reg!=0: x = Dropout(drop_in, name='drop0')(Inputs)
 else: x = Dense(num_unit,name='dense1',activation='relu')(Inputs)
 for i in range(1,num_hidden):
   x = Dense(num_unit,activation='relu',name='dense{}'.format(i+1))(x)
   if l1Reg!=0: x = Dropout(drop_hidden, name='drop{}'.format(i+1))(x)
 x = Dense(nclasses, name='dense')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def float_model_hinge_constraint(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0,h5fName=None):
    
 print("RETRAIN FLOAT MNIST MODEL WITH DROPPED WEIGHTS")
 h5f = h5py.File(h5fName)

 x = Dense(num_unit,name='dense0',W_regularizer=l1(l1Reg),kernel_constraint=zero_some_weights(binary_tensor=h5f['dense0'][()].tolist()))(Inputs)
 for i in range(num_hidden):
   x = Dense(num_unit,activation='relu',name='dense{}'.format(i+1),W_regularizer=l1(l1Reg),kernel_constraint=zero_some_weights(binary_tensor=h5f['dense{}'.format(i+1)][()].tolist()))(x)
 x = Dense(nclasses,name='dense',kernel_constraint=zero_some_weights(binary_tensor=h5f['dense'][()].tolist()))(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def float_model_hinge_batchnorm(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING FLOAT MNIST MODEL WITH",num_unit,"NEURONS AND",num_hidden,"LAYERS")

 if l1Reg!=0: x = Dropout(drop_in, name='drop0')(Inputs)
 else: x = Dense(num_unit,name='dense0',W_regularizer=l1(l1Reg))(Inputs)
 for i in range(num_hidden):
   x = Dense(num_unit,name='dense{}'.format(i+1),W_regularizer=l1(l1Reg))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation('relu', name='act{}'.format(i+1))(x)
   if l1Reg!=0: x = Dropout(drop_hidden, name='drop{}'.format(i+1))(x)
 x = Dense(nclasses, name='dense')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def float_model_hinge_batchnorm_constraint(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0,h5fName=None):
    
 print("RETRAIN FLOAT MNIST MODEL WITH DROPPED WEIGHTS")
 h5f = h5py.File(h5fName)

 x = Dense(num_unit,name='dense0',W_regularizer=l1(l1Reg),kernel_constraint=zero_some_weights(binary_tensor=h5f['dense0'][()].tolist()))(Inputs)
 for i in range(num_hidden):
   x = Dense(num_unit,name='dense{}'.format(i+1),W_regularizer=l1(l1Reg),kernel_constraint=zero_some_weights(binary_tensor=h5f['dense{}'.format(i+1)][()].tolist()))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation('relu', name='act{}'.format(i+1))(x)
 x = Dense(nclasses,name='dense',kernel_constraint=zero_some_weights(binary_tensor=h5f['dense'][()].tolist()))(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def binary_model_binarytanh(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING BINARY MNIST MODEL WITH BINARY TANH,",num_unit,"NEURONS AND",num_hidden,"LAYERS")
     
 x = DropoutNoScaleForBinary(drop_in,name='drop0')(Inputs)
 for i in range(num_hidden):
   x = BinaryDense(num_unit, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense{}'.format(i+1))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation(binary_tanh, name='act{}'.format(i+1))(x)
   x = DropoutNoScaleForBinary(drop_hidden, name='drop{}'.format(i+1))(x)
 x = BinaryDense(nclasses, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def binary_model_relu(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING BINARY MNIST MODEL WITH RELU,",num_unit,"NEURONS AND",num_hidden,"LAYERS")

 x = DropoutNoScaleForBinary(drop_in,name='drop0')(Inputs)
 for i in range(num_hidden):
   x = BinaryDense(num_unit, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense{}'.format(i+1))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation('relu', name='act{}'.format(i+1))(x)
   x = DropoutNoScaleForBinary(drop_hidden, name='drop{}'.format(i+1))(x)
 x = BinaryDense(nclasses, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def binary_model_maxrelu(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING BINARY MNIST MODEL WITH MAXRELU1,",num_unit,"NEURONS AND",num_hidden,"LAYERS")

 x = DropoutNoScaleForBinary(drop_in,name='drop0')(Inputs)
 for i in range(num_hidden):
   x = BinaryDense(num_unit, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense{}'.format(i+1))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation(relu1, name='act{}'.format(i+1))(x)
   x = DropoutNoScaleForBinary(drop_hidden, name='drop{}'.format(i+1))(x)
 x = BinaryDense(nclasses, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def ternary_model_ternarytanh(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING TERNARY MNIST MODEL WITH TERNARY TANH,",num_unit,"NEURONS AND",num_hidden,"LAYERS")

 x = DropoutNoScaleForTernary(drop_in,name='drop0')(Inputs)
 for i in range(num_hidden):
   x = TernaryDense(num_unit, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense{}'.format(i+1))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation(ternary_tanh, name='act{}'.format(i+1))(x)
   x = DropoutNoScaleForTernary(drop_hidden, name='drop{}'.format(i+1))(x)
 x = TernaryDense(nclasses, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def ternary_model_relu(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING TERNARY MNIST MODEL WITH RELU,",num_unit,"NEURONS AND",num_hidden,"LAYERS")

 x = DropoutNoScaleForTernary(drop_in,name='drop0')(Inputs)
 for i in range(num_hidden):
   x = TernaryDense(num_unit, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense{}'.format(i+1))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation('relu', name='act{}'.format(i+1))(x)
   x = DropoutNoScaleForTernary(drop_hidden, name='drop{}'.format(i+1))(x)
 x = TernaryDense(nclasses, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def ternary_model_maxrelu(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):
    
 print("CREATING TERNARY MNIST MODEL WITH MAXRELU,",num_unit,"NEURONS AND",num_hidden,"LAYERS")

 x = DropoutNoScaleForTernary(drop_in,name='drop0')(Inputs)
 for i in range(num_hidden):
   x = TernaryDense(num_unit, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense{}'.format(i+1))(x)
   x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn{}'.format(i+1))(x)
   x = Activation(relu1, name='act{}'.format(i+1))(x)
   x = DropoutNoScaleForTernary(drop_hidden, name='drop{}'.format(i+1))(x)
 x = TernaryDense(nclasses, H=1.0, kernel_lr_multiplier=1.0, use_bias=False,name='dense')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def float_cnn_model(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):

 print("CREATING FLOAT MNIST MODEL WITH CNN")
 
 x = Conv2D(8, kernel_size=(3, 3),activation='relu',name='conv0')(Inputs) #32
 x = Conv2D(16, (3, 3), activation='relu', name='conv1')(x) #64
 x = MaxPooling2D(pool_size=(2, 2),name='mp1')(x)
 x = Dropout(0.25,name='drop1')(x)
 x = Flatten()(x)
 x = Dense(128, activation='relu',name='dense')(x)
 x = Dropout(0.5,name='drop')(x)
 predictions = Dense(nclasses, activation='softmax',name='output')(x)
 #model.add(Dense(num_classes))
 #model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn'))
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def float_cnn_model_hinge(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):

 print("CREATING FLOAT MNIST MODEL WITH CNN AND HINGE LOSS")

 x = Conv2D(32, kernel_size=(3, 3),activation='relu',name='conv0')(Inputs)
 x = Conv2D(64, (3, 3), activation='relu', name='conv1')(x)
 x = MaxPooling2D(pool_size=(2, 2),name='mp1')(x)
 x = Dropout(0.25,name='drop1')(x)
 x = Flatten()(x)
 x = Dense(128, activation='relu',name='dense1')(x)
 x = Dropout(0.5,name='drop')(x)
 x = Dense(nclasses, name='dense2')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

def binary_cnn_model(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):

 print("CREATING BINARY MNIST MODEL WITH CNN AND HINGE LOSS")

 #conv1
 x = BinaryConv2D(8, kernel_size=(3, 3),
                       data_format='channels_first',
                       H=1.0, kernel_lr_multiplier=1.0,
                       padding='same', use_bias=False, name='conv1')(Inputs)
 x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1')(x)
 x = Activation(binary_tanh, name='act1')(x)
 #conv2
 x = BinaryConv2D(16, kernel_size=(3, 3), H=1.0, kernel_lr_multiplier=1.0,
                       data_format='channels_first',
                       padding='same', use_bias=False, name='conv2')(x)
 x = MaxPooling2D(pool_size=(2, 2), name='pool2', data_format='channels_first')(x)
 x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2')(x)
 x = Activation(binary_tanh, name='act2')(x)
 x = Flatten()(x)
 # dense1
 x = BinaryDense(128, H=1.0, kernel_lr_multiplier=1.0, use_bias=False, name='dense3')(x)
 x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn3')(x)
 x = Activation(binary_tanh, name='act3')(x)
 # dense2
 x = BinaryDense(nclasses, H=1.0, kernel_lr_multiplier=1.0, use_bias=False, name='dense4')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn4')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model

'''
def binary_cnn_model(Inputs,num_unit,num_hidden,nclasses,drop_in,drop_hidden,epsilon,momentum,l1Reg=0):

 print("CREATING BINARY MNIST MODEL WITH CNN AND HINGE LOSS")

 #conv1
 x = BinaryConv2D(128, kernel_size=(3, 3),
                      data_format='channels_first',
                      H=1.0, kernel_lr_multiplier=1.0,
                      padding='same', use_bias=False, name='conv1')(Inputs)
 x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1')(x)
 x = Activation(binary_tanh, name='act1')(x)
 #conv2
 x = BinaryConv2D(128, kernel_size=(3, 3), H=1.0, kernel_lr_multiplier=1.0,
                      data_format='channels_first',
                      padding='same', use_bias=False, name='conv2')(x)
 x = MaxPooling2D(pool_size=(2, 2), name='pool2', data_format='channels_first')(x)
 x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2')(x)
 x = Activation(binary_tanh, name='act2')(x)
 # conv3
 x = BinaryConv2D(256, kernel_size=(3,3), H=1.0, kernel_lr_multiplier=1.0,
                      data_format='channels_first',
                      padding='same', use_bias=False, name='conv3')(x)
 x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3')(x)
 x = Activation(binary_tanh, name='act3')(x)
 # conv4
 x = BinaryConv2D(256, kernel_size=(3,3), H=1.0, kernel_lr_multiplier=1.0,
                      data_format='channels_first',
                      padding='same', use_bias=False, name='conv4')(x)
 x = MaxPooling2D(pool_size=(2,2), name='pool4', data_format='channels_first')(x)
 x = BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4')(x)
 x = Activation(binary_tanh, name='act4')(x)
 x = Flatten()(x)
 # dense1
 x = BinaryDense(1024, H=1.0, kernel_lr_multiplier=1.0, use_bias=False, name='dense5')(x)
 x = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5')(x)
 x = Activation(binary_tanh, name='act5')(x)
 # dense2
 x = BinaryDense(nclasses, H=1.0, kernel_lr_multiplier=1.0, use_bias=False, name='dense6')(x)
 predictions = BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6')(x)
 model = Model(inputs=Inputs, outputs=predictions)
 return model
'''
