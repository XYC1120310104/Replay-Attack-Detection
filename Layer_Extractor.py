# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:54:10 2017

@author: Leon
"""

from keras.preprocessing import image
from keras.optimizers import SGD
import os, sys
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,load_model,Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.initializers import he_normal,glorot_normal
from keras import utils
num_classes =2
batch_size =64
epochs =20

best_model_file = './parameter/GlobalAvergePolling_lcnn_blackman_frames400_adam_epochs25_batch32.h5'

def featProcess():
    #listWav.getData(FrameLen=256,FrameInc=64,NFFT=512)
    
    trainData = np.load('/home/xuyongchao/CNNnetwork/feature/trainData.npy')
    trainLabel = np.load('feature/trainLabel.npy')
    
    devData = np.load('feature/devData.npy')
    devLabel = np.load('feature/devLabel.npy')

    evalData = np.load('feature/evalData.npy')
    evalLabel = np.load('feature/evalLabel.npy')
    
    img_rows, img_cols =trainData[0].shape
    
    x_train = trainData.reshape(trainData.shape[0], img_rows, img_cols, 1)
    y_train = utils.to_categorical(trainLabel,num_classes)

    x_dev = devData.reshape(devData.shape[0], img_rows, img_cols, 1)
    y_dev = utils.to_categorical(devLabel,num_classes)

    x_test = evalData.reshape(evalData.shape[0], img_rows, img_cols, 1)
    y_test = utils.to_categorical(evalLabel,num_classes)

    return x_train,trainLabel,x_dev,devLabel,x_test,evalLabel
    
x_train,y_train,x_dev,y_dev,x_test,y_test= featProcess()

def buildModel():
    cnn = Sequential()

    cnn.add(Conv2D(32, (5,5), strides=(1,1), padding='same',kernel_initializer=glorot_normal(), input_shape=x_train.shape[1:],name='conv1_1'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(Conv2D(32, (3,3),strides=(1,1),padding='same',kernel_initializer=glorot_normal(),name='conv2_1'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))	
    cnn.add(Conv2D(48,(3,3),strides=(1,1),padding='same',kernel_initializer=glorot_normal(),name='conv2_2'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))	
    cnn.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(Conv2D(48, (3,3), strides=(1,1),padding='same',kernel_initializer=glorot_normal(),name='conv3_1'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv2D(64, (3,3), strides=(1,1),padding='same',kernel_initializer=glorot_normal(),name='conv3_2'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(Conv2D(64, (3,3), strides=(1,1),padding='same',kernel_initializer=glorot_normal(),name='conv4_1'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv2D(32, (3,3), strides=(1,1),padding='same',kernel_initializer=glorot_normal(),name='conv4_2'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(Conv2D(32, (3,3), strides=(1,1),padding='same',kernel_initializer=glorot_normal(),name='conv5_1'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv2D(32, (3,3), strides=(1,1),padding='same',kernel_initializer=glorot_normal(),name='conv5_2'))
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    cnn.add(Flatten())

    cnn.add(Dense(128,name='Dense_1')) # 4096
    #cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))

    cnn.add(Dropout(0.5))

    cnn.add(Dense(num_classes,name='Dense_2'))
    cnn.add(Activation('softmax'))
    return cnn
#model = load_model('./parameter/My_lightcnn_frame400_blackman_epoch5_retrain_PowSpec.h5')   
#model = buildModel()
#model.load_weights('./parameter/VGG_frames400_epochs25.h5', by_name=True)
model =load_model(best_model_file)
cnn_convolutional_only = Model(input=model.input, output=model.get_layer('GAPooling').output) # TRUNCATE MODEL AT DENSE LAYER

#Extract Data
trainData = cnn_convolutional_only.predict(x_train)
devData = cnn_convolutional_only.predict(x_dev)
testData = cnn_convolutional_only.predict(x_test)
trainSet = np.c_[y_train,trainData]
devSet = np.c_[y_dev,devData]
testSet = np.c_[y_test,testData]
np.savetxt('tr_dat.csv',trainSet,delimiter=',')
np.savetxt('dev_dat.csv',devSet,delimiter=',')
np.savetxt('te_dat.csv',testSet,delimiter=',')
