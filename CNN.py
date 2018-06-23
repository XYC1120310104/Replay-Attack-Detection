# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:44:08 2017

@author: Leon
"""

#import wechat_utils
#import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Input,Sequential,load_model,Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda, add,concatenate
#GlobalAveragePooling2D,PReLU
from keras.initializers import he_normal,glorot_normal,glorot_uniform

import keras.backend as K
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from keras import metrics

import listWav
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

if('tensorflow' == K.backend):
	#import tensorflow as tf
	from keras.backend.tensorflow_backend import set_session
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

#wechat_utils.login()

num_classes = 2
batch_size = 32
epochs = 25
lr=0.0001
patience_iter = 1
best_model_file = './parameter/LightCnn_blackman_frames400_sgd_epochs25_batch32.h5'

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def featProcess():
    #listWav.getData(img_rows=864,FrameLen=400,FrameInc=100,NFFT=800,winfunc=np.blackman)
    
    trainData = np.load('/home/xuyongchao/CNNnetwork/feature/trainData.npy')
    trainLabel = np.load('feature/trainLabel.npy')
    
    devData = np.load('feature/devData.npy')
    devLabel = np.load('feature/devLabel.npy')

    evalData = np.load('feature/evalData.npy')
    evalLabel = np.load('feature/evalLabel.npy')
    
    #trainData = np.concatenate((trainData,devData))
    #trainLabel = np.concatenate((trainLabel,devLabel)) 
    
    img_rows, img_cols =trainData[0].shape
    
    x_train = trainData.reshape(trainData.shape[0], img_rows, img_cols, 1)
    y_train = keras.utils.to_categorical(trainLabel,num_classes)
    #y_train = trainLabel
    
    x_dev = devData.reshape(devData.shape[0], img_rows, img_cols, 1)
    y_dev = keras.utils.to_categorical(devLabel,num_classes)
    #y_dev = devLabel
    
    x_test = evalData.reshape(evalData.shape[0], img_rows, img_cols, 1)
    y_test = keras.utils.to_categorical(evalLabel,num_classes)
    #y_test = evalLabel
    
 
    return x_train,y_train,x_dev,y_dev,devLabel,x_test,y_test,evalLabel
    
x_train,y_train,x_dev,y_dev,devLabel,x_test,y_test,evalLabel= featProcess()

def EER_score(y,y_pred_prob,pos_label=1):
    fpr,tpr,thresholds = roc_curve(y,y_pred_prob)
    fnr = 1 - tpr
    EER_fpr = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.nanargmin(np.absolute((fnr-fpr)))]
    EER = (EER_fpr+EER_fnr)/2
    return EER

def auc(y_true,y_pred):
    
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0,1,1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0,1,1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)),pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s,axis=0)

def binary_PFA(y_true,y_pred,threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold,'float32')
    N = K.sum(1-y_true)
    FP = K.sum(y_pred-y_pred*y_true)
    return FP/N

def binary_PTA(y_true,y_pred,threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold,'float32')
    P = K.sum(1-y_true)
    TP = K.sum(y_pred-y_pred*y_true)
    return TP/P 
 
def CalEER(y,y_pred_prob):
    fpr,tpr,thresholds = roc_curve(y,y_pred_prob)
    fnr = 1 - tpr
    eer_thresholds = thresholds[np.nanargmin(np.absolute((fnr-fpr)))]
    EER_fpr = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.nanargmin(np.absolute((fnr-fpr)))]
    EER = (EER_fpr+EER_fnr)/2
    print "EER_fpr:",EER_fpr
    print "EER_fnr:",EER_fnr
    print "EER:",EER
    plt.plot(fpr,tpr,color="red")
    plt.plot(np.arange(0.0,1.0,0.01),1.0-np.arange(0.0,1.0,0.01),color="blue")
    plt.annotate(EER,xy=(EER,1-EER),xytext=(0.1+EER,1-EER),
                 arrowprops=dict(arrowstyle='->',connectionstyle='arc3'))
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.title("ROC curve for diabetes classifier")
    plt.xlabel("False Positive Rate (1-Specificity)")
    plt.ylabel("True positive Rate (Sensitivity)")
    plt.grid(True)
    plt.show()
    return eer_thresholds

def analysisModel(y,y_pred):
    print classification_report(y,y_pred)
    confusion = confusion_matrix(y,y_pred)
    TP = confusion[1,1]
    FP = confusion[0,1]
    TN = confusion[0,0]
    FN = confusion[1,0]
    print "TP:",TP,"FN:",FNms
    print "Recall:",recall_score(y,y_pred)#TPR,sensitivity
    print "Specificity:",TN / float(TN+FP)#TNR
    print "1-Specificity",FP / float(TN+FP)#FPR
    print "Precision:",precision_score(y,y_pred)
    
#Loss History
    
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self,logs={}):
        self.losses = []
    def on_batch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))

#add EER metric to Keras

class EERMetricCallback(keras.callbacks.Callback):
    def __init__(self,predict_batch_size=1024,include_on_batch=False):
        super(EERMetricCallback,self).__init__()
        self.predict_batch_size=predict_batch_size
        self.include_on_batch=include_on_batch
        
    def on_batch_begin(self,batch,logs={}):
        pass
    
    def on_batch_end(self,batch,logs={}):
        pass
        '''
        if (self.include_on_batch):
            logs['eer_val']=float('-inf')
            if (self.validation_data):
                logs['eer_val']=EER_score(self.validation_data[1],
                    self.model.predict(self.validation_data[0],batch_size=self.predict_batch_size))
        '''       
    def on_train_begin(self,logs={}):
        if not ('eer_val' in self.params['metrics']):
            self.params['metrics'].append('eer_val')
            
    def on_train_end(self,logs={}):
        pass
    
    def on_epoch_begin(self,epoch,logs={}):
        pass
    
    def on_epoch_end(self,epoch,logs={}):
        logs['eer_val']=float('-inf')
        if (self.validation_data):
            logs['eer_val']=EER_score(self.validation_data[1][:,1],
                self.model.predict(self.validation_data[0],batch_size=self.predict_batch_size)[:,1])
            
def lr_scheduler(epoch, lr_base=lr, lr_power=0.9, mode='power_decay'):
    if mode == 'power_decay':
        lr = lr_base * ((1-float(epoch)/epochs)**lr_power)
    if mode == 'progressive_drops':
        if epoch >= 0.9*epochs:
            lr = 0.00003
        elif epoch >= 0.7*epochs:
            lr = 0.00007
        elif epoch >= 0.5*epochs:
            lr = 0.0001
        elif epoch >= 0.3*epochs:
            lr = 0.0003
        elif epoch >= 0.2*epochs:
            lr = 0.006
        elif epoch >= 0.1*epochs:
            lr = 0.008
        else:
            lr = 0.01
    print('lr is %f' % lr)
    return lr
        
history = LossHistory()
eer_metric = EERMetricCallback(predict_batch_size=batch_size)

cb=[
    history, 
    eer_metric,
    #wechat_utils.sendmessage(savelog=True,fexten='TEST'),
    ModelCheckpoint(best_model_file,monitor='eer_val',verbose=2,save_best_only=True,mode='min'),
    LearningRateScheduler(lr_scheduler),
    #EarlyStopping(monitor='eer_val',patience=patience_iter,verbose=2,mode='min')
    ]


def MFM(x):
	channels = x.shape[-1]
	ret = K.maximum(x[...,0:channels/2],x[...,channels/2:])
	return ret 

def MFM_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 4
	shape[-1] = shape[-1]/2
	return tuple(shape)

def buildLcnn():
	input_tensor = Input(shape=x_train.shape[1:],name='input')
	x = Conv2D(32, (5,5), strides=(1,1), kernel_initializer=glorot_normal(),padding='same')(input_tensor)
	x = Lambda(MFM)(x)
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

	x = Conv2D(32, (1,1),strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
	x = Lambda(MFM)(x)
	x = Conv2D(64,(3,3),strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
	x = Lambda(MFM)(x)
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)

	x = Conv2D(64, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
	x = Lambda(MFM)(x)
	x = Conv2D(96, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
	x = Lambda(MFM)(x) 
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)	
	
	x = Conv2D(96, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
	x = Lambda(MFM)(x)
	x = Conv2D(64, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
	x = Lambda(MFM)(x)
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)	

	x = Conv2D(64, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
	x = Lambda(MFM)(x)
	x = Conv2D(64, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x) 
	x = Lambda(MFM)(x)
	x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)	

	x = Flatten()(x)
	x = Dropout(0.3)(x)
	x = Dense(64,activation='relu')(x)
	#x = Dropout(0.5)(x)
	x = Dense(num_classes,activation='softmax',name='output')(x)
	
	model = Model(input_tensor,x)

	return model 

def block_unit( ):
    input_tensor = Input(shape=x_train.shape[1:],name='input')
    x = Conv2D(16, (5,5), strides=(1,1), kernel_initializer=glorot_normal(),padding='same')(input_tensor)
    x = Lambda(MFM)(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    
    y = Conv2D(16, (1,1),strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    y = Lambda(MFM)(y)
    y = Conv2D(24,(3,3),strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(y)
    y = Lambda(MFM)(y)
    x = concatenate([x,y])
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    
    y = Conv2D(24, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    y = Lambda(MFM)(x)
    x = Conv2D(32, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Lambda(MFM)(x)
    x = concatenate([x,y])
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)	
    
    y = Conv2D(32, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    y = Lambda(MFM)(y)
    y = Conv2D(16, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(y)
    y = Lambda(MFM)(y)
    x = concatenate([x,y])
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)	
    
    y = Conv2D(16, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    y = Lambda(MFM)(y)
    y = Conv2D(16, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(y) 
    y = Lambda(MFM)(y)
    x = concatenate([x,y])
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
	
    x = Flatten()(x)
    x = Dropout(0.7)(x)
    x = Dense(64,activation='relu')(x)
    x = Dense(num_classes,activation='softmax',name='output')(x)
    
    model = Model(input_tensor,x)    
    return model

def buildNIN():
    input_tensor = Input(shape=x_train.shape[1:],name='input')
    x = Conv2D(64, (3,3), strides=(1,1), kernel_initializer=glorot_normal(),padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(64, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    
    x = Conv2D(64,(3,3),strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    
    x = Conv2D(8,(3,3),strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(4, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Activation('relu')(x)    
    x = Conv2D(2, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same')(x)
    x = Activation('relu')(x)   
    
    x = GlobalAveragePooling2D(name='GAP')(x)
    x = Activation('softmax')(x)
    
    model = Model(input_tensor,x)
    return model
    
def buildModel():
	input_tensor = Input(shape=x_train.shape[1:],name='input')
	'''
	x = block_unit(input_tensor)
	y = block_unit(input_tensor)
	catenate = add([x,y])'''
    
	y = Conv2D(64, (5,5), strides=(1,1), kernel_initializer=glorot_normal(),padding='same', name='conv1_1')(input_tensor)
	y = Lambda(MFM,name='MFM1_1')(y)
	y = MaxPooling2D(pool_size=(2,2),strides=(2,2))(y)

	y = Conv2D(64, (1,1),strides=(1,1),kernel_initializer=glorot_normal(),padding='same',name='conv2_1')(y)
	y = Lambda(MFM,name='MFM2_1')(y)
	y = Conv2D(128,(3,3),strides=(1,1),kernel_initializer=glorot_normal(),padding='same',name='conv2_2')(y)
	y = Lambda(MFM,name='MFM2_2')(y)
	y = MaxPooling2D(pool_size=(2,2),strides=(2,2))(y)

	y = Conv2D(128, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same',name='conv3_1')(y)
	y = Lambda(MFM,name='MFM3_1')(y)
	y = Conv2D(256, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same',name='conv3_2')(y)
	y = Lambda(MFM,name='MFM3_2')(y) 
	y = MaxPooling2D(pool_size=(2,2),strides=(2,2))(y)	
	
	y = Conv2D(256, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same',name='conv4_1')(y)
	y = Lambda(MFM,name='MFM4_1')(y)
	y = Conv2D(128, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same',name='conv4_2')(y)
	y = Lambda(MFM,name='MFM4_2')(y)
	y = MaxPooling2D(pool_size=(2,2),strides=(2,2))(y)	

	y = Conv2D(128, (1,1), strides=(1,1),kernel_initializer=glorot_normal(),padding='same',name='conv5_1')(y)
	y = Lambda(MFM,name='MFM5_1')(y)
	y = Conv2D(128, (3,3), strides=(1,1),kernel_initializer=glorot_normal(),padding='same',name='conv5_2')(y) 
	
    #y = Lambda(MFM,name='MFM5_2')(y)
    #y = Activation('linear')(y)
    #y = PReLU()(y)
	#y = MaxPooling2D(pool_size=(2,2),strides=(2,2))(y)	
    
 	y = GlobalAveragePooling2D(name='GAPooling')(y)
	y = Dense(num_classes,activation='softmax',name='output')(y)
	
	model = Model(input_tensor,y)

	return model 
    
def trainModel(model):
    sgd = keras.optimizers.SGD(lr=lr,momentum=0.9,nesterov=True)
    adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.9, epsilon=1e-6)
    adam = keras.optimizers.Adam(lr=3e-5,beta_1=0.9,beta_2=0.999,epsilon=1e-8) 
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    
    nb_train_samples = x_train.shape[0]
    
    train_datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    	# Compute quantities required for feature-wise normalization
    	# (std, mean, and principal components if ZCA whitening is applied).
    
    train_datagen.fit(x_train)
    train_generator = train_datagen.flow(x_train, y_train,batch_size=batch_size)
    
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(train_generator,
                            steps_per_epoch=int(np.ceil(nb_train_samples / float(batch_size))),
                            epochs=epochs,
                            validation_data=(x_test,y_test),
                            verbose=2,
                            callbacks=cb,
                            workers=4)
    
    #model.save('./parameter/lcnn_blackman_frames400_adadelta_epochs20.h5', model)
    return model

#model = load_model('./9.15_My_lightcnn_frame400_blackman_epoch5_retrain_PowSpec.h5')
model = block_unit()

print model.summary()
from keras.utils import plot_model
plot_model(model,to_file='model.png')
model = trainModel(model)
#callback print
    
#print(history.losses)

#dev
print '********************Scores on devSet********************'
#dev_yPred = model.predict_classes(x_dev)
dev_prediction = model.predict(x_dev)#prob of classes
#dev_score = model.evaluate(x_dev, y_dev, verbose=1)

CalEER(devLabel,dev_prediction[:,1])
#analysisModel(devLabel,dev_yPred)

#print('dev loss:', dev_score[0])
#print('dev accuracy:', dev_score[1])
#test
print '********************Scores on testSet*********************'
#yPred = model.predict_classes(x_test)
prediction = model.predict(x_test)#prob of classes
#score = model.evaluate(x_test, y_test, verbose=1)

CalEER(evalLabel,prediction[:,1])
#analysisModel(evalLabel,yPred)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
