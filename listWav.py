#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:49:13 2017

@author: gavin
"""

import os
import sys

import numpy as np
import scipy.io.wavfile as wav
from sklearn.preprocessing import StandardScaler
from sigprocess import audio2frame
from sigprocess import log_spectrum_power,spectrum_power
from calcmfcc import calcMFCC_delta_delta,log_fbank

wav_path = '../wav/'
proctol_path = '../protocol/ASVspoof2017_'


def get_wav_files(wav_path,proctol_path,dataSet):
    wav_path = wav_path + dataSet
    X = []
    Y = []
    label_map = {'genuine':1,'spoof':0}
    label_file = proctol_path + dataSet +'.trl'
    with open(label_file,'rb+') as file:
        for line in file.readlines():
            tmp = line.split()
            Y.append(label_map[tmp[1]])
            filename_path = os.sep.join([wav_path,tmp[0]])
            X.append(filename_path)
    return X,Y

def getDataSet(wav_files,label,img_rows=864,FrameLen=256,FrameInc=64,NFFT=800,winfunc=lambda x:np.ones((x,))):
    X = []
    Y = []
    for i,singleWav in enumerate(wav_files):
        (rate,sig) = wav.read(singleWav)
        frames = audio2frame(sig,FrameLen,FrameInc,winfunc)#blackman window alpha=0.16
        logPowSpec = log_spectrum_power(frames,NFFT,norm=1)
        #logPowSpec = spectrum_power(frames,NFFT)
        #logPowSpec = log_fbank(sig,win_length=0.025,win_step=0.01,filters_num=39,NFFT)
        #logPowSpec = calcMFCC_delta_delta(sig,win_length=0.025,win_step=0.01,cep_num=20,filters_num=26,NFFT)
        dim1,dim2 = logPowSpec.shape
        if dim1 >= img_rows :
            divNum = dim1 / img_rows
            logPowSpec = logPowSpec[0:divNum*img_rows,:]
            logPowSpec = logPowSpec.reshape(divNum,img_rows,-1)
            X.extend(logPowSpec)
            Y.extend([label[i]]*divNum)
        else :
            divNum = img_rows / dim1
            mod = img_rows % dim1
            logPowSpec =  np.array([logPowSpec]*divNum).reshape((divNum*dim1,dim2))
            logPowSpec =  np.concatenate((logPowSpec,logPowSpec[0:mod,:]))
            X.append(logPowSpec)
            Y.append(label[i])
    return X,Y

def DataProcess(dataset,img_rows=256,FrameLen=256,FrameInc=64,NFFT=800,winfunc=lambda x:np.ones((x,))):
    wav_files,label = get_wav_files(wav_path,proctol_path,dataset)
    X,label = getDataSet(wav_files,label,img_rows,FrameLen,FrameInc,NFFT,winfunc)
    X = [ (i - np.mean(i,axis=0,dtype=float))/np.std(i,axis=0,dtype=float) for i in X ]#X = [ StandardScaler().fit_transform(i) for i in X ]
    dim = [i.shape[0] for i in X]
    return X,label,dim

def getData(img_rows=256,FrameLen=256,FrameInc=64,NFFT=800,winfunc=lambda x:np.ones((x,))):
    print "Extract Feature from TrainSet:"
    trainData,trainLabel,dim_T = DataProcess('train',img_rows,FrameLen,FrameInc,NFFT,winfunc)
    print "Ending from TrainSet"
    print "Extraxt Feature from DevSet"
    devData,devLabel,dim_D = DataProcess('dev',img_rows,FrameLen,FrameInc,NFFT,winfunc)
    print "Ending from DevSet"
    print "Extract Feature from TestSet"
    evalData,evalLabel,dim_E = DataProcess('eval',img_rows,FrameLen,FrameInc,NFFT,winfunc)
    print "Ending from TestSet"
    np.save('feature/trainData',np.array(trainData))
    np.save('feature/devData',np.array(devData))
    np.save('feature/evalData',np.array(evalData))
    np.save('feature/trainLabel',np.array(trainLabel))
    np.save('feature/devLabel',np.array(devLabel))
    np.save('feature/evalLabel',np.array(evalLabel))



























