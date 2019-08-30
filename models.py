from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, Clip, DropoutNoScaleForBinary
from ternary_ops import ternarize
from ternary_layers import TernaryDense, DropoutNoScaleForTernary
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
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
 else: x = Dense(num_unit,name='dense0',W_regularizer=l1(l1Reg))(Inputs)
 for i in range(num_hidden):
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
 else: x = Dense(num_unit,name='dense0',W_regularizer=l1(l1Reg))(Inputs)
 for i in range(num_hidden):
   x = Dense(num_unit,activation='relu',name='dense{}'.format(i+1),W_regularizer=l1(l1Reg))(x)
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
