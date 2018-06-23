# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:01:05 2017

@author: Leon
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
 
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import make_scorer,roc_curve,auc
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,roc_auc_score)
from sklearn.metrics import (mean_squared_error,mean_absolute_error)
   
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV,train_test_split,StratifiedKFold,StratifiedShuffleSplit,cross_val_score

from sklearn.externals import joblib
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin,ClassifierMixin

class PseudoLabeler(BaseEstimator, ClassifierMixin):
	def __init__(self, model, unlabled_data, sample_rate=0.2, seed=42):
	     assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'
	     self.sample_rate = sample_rate
	     self.seed = seed
	     self.model = model
	     self.model.seed = seed
	     self.unlabled_data = unlabled_data

	def get_params(self, deep=True):
	     return {
	     "sample_rate": self.sample_rate,
	     "seed": self.seed,
	     "model": self.model,
	     "unlabled_data": self.unlabled_data,
	     }

	def set_params(self, **parameters):
	     for parameter, value in parameters.items():
		setattr(self, parameter, value)
	     return self

	def fit(self, X, y):
	     augemented_train = self.__create_augmented_train(X, y)
	     self.model.fit(
	     augemented_train[:,:-1],
	     augemented_train[:,-1],
	     )
	     return self

	def __create_augmented_train(self, X, y):
	     totalNum = self.unlabled_data.shape[0]
	     num_of_samples = int(totalNum * self.sample_rate)
	     idx = np.random.randint(totalNum,size=num_of_samples)
	     # Train the model and creat the pseudo-labels
	     self.model.fit(X, y)
	     pseudo_label = self.model.predict(self.unlabled_data)
	     pseudo_data = self.unlabled_data
         
	     sampled_pseudo_data = pseudo_data[idx,:]
	     sampled_pseudo_label = pseudo_label[idx]
	     
	     augemented_train_data = np.vstack((X,sampled_pseudo_data))
	     augemented_train_label = np.hstack((y,sampled_pseudo_label))
	     augemented_train = np.hstack((augemented_train_data,augemented_train_label.reshape(-1,1)))
	     return shuffle(augemented_train)

	def predict(self, X):
	    return self.model.predict(X)

	def predict_proba(self,X):
	    return self.model.predict_proba(X)

	def get_model_name(self):
	    return self.model.__class__.__name__


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
    
    return EER,eer_thresholds
def analysisModel(expected,predicted):
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average="binary")
    precision = precision_score(expected, predicted , average="binary")
    f1 = f1_score(expected, predicted , average="binary")
    cm = confusion_matrix(expected, predicted)
    TP = cm[1,1]
    FP = cm[0,1]
    TN = cm[0,0]
    FN = cm[1,0]
    print "TP:",TP,"FP:",FP,"TN:",TN,"FN:",FN
    tpr = TP / float(TP+FN)
    fpr = FP / float(TN+FP)
    tnr = TN / float(TN+FP)
    print("fpr","%.3f" %fpr)
    print("tpr","%.3f" %tpr)
    print ("TNR|Specificity:",tnr)#TNR
    print ("FPR|1-Specificity",fpr)#FPR
    print("Accuracy:","%.3f" %accuracy)
    print("precision:","%.3f" %precision)
    print("recall","%.3f" %recall)
    print("f-score","%.3f" %f1)

def EER_score(fpr,tpr):
    fnr = 1 - tpr
    EER_fpr = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.nanargmin(np.absolute((fnr-fpr)))]
    EER = (EER_fpr+EER_fnr)/2
    return EER

def custom_auc(ground_truth,predictions):
    fpr,tpr,_ = roc_curve(ground_truth,predictions[:,1],pos_label=1)
    return EER_score(fpr,tpr)

scoring_fnc = make_scorer(custom_auc,greater_is_better=True,needs_proba=True)    

traindata = pd.read_csv('./data/tr_dat.csv', header=None)
devdata = pd.read_csv('./data/dev_dat.csv',header=None)
testdata = pd.read_csv('./data/te_dat.csv', header=None)

num_classes = 2
fdim = 128
X = traindata.iloc[:,1:fdim]
Y = traindata.iloc[:,0]

D = devdata.iloc[:,1:fdim]
L = devdata.iloc[:,0]

C = testdata.iloc[:,0]
T = testdata.iloc[:,1:fdim]

# scaler = Normalizer().fit(X)
# trainX = scaler.transform(X)

# scaler = Normalizer().fit(T)
# testT = scaler.transform(T)
# print(testT)

traindata = np.array(X)
trainlabel = np.array(Y)
devdata = np.array(D)
devlabel = np.array(L)
#traindata = np.concatenate((traindata,devdata))
#trainlabel = np.concatenate((trainlabel,devlabel)) 
print(traindata.shape)

testdata = np.array(T)
testlabel = np.array(C)
print(testdata.shape)

expected = testlabel

#GMM/single Training 10.15%
print("******************************GMM******************************")
genuineTrainDataIndex = np.array(np.where(trainlabel==1)).reshape(-1)
spoofTrainDataIndex = np.array(np.where(trainlabel==0)).reshape(-1)
genuineTrainData = traindata[genuineTrainDataIndex,:]
spoofTrainData = traindata[spoofTrainDataIndex,:]

genuineModel = GaussianMixture(n_components=425,covariance_type='full',init_params = 'kmeans', max_iter = 50,random_state=0)
spoofModel = GaussianMixture(n_components=425,covariance_type='full',init_params = 'kmeans', max_iter = 50,random_state=0)

#genuine  
genuineModel.fit(genuineTrainData)  
#spoof 
spoofModel.fit(spoofTrainData)
joblib.dump(genuineModel,'./ML_model_parameters/genuineGMM.pkl')
joblib.dump(spoofModel,'./ML_model_parameters/spoofGMM.pkl')
# dev predictions
dev_genuine_prob = genuineModel.score_samples(devdata)
dev_spoof_prob = spoofModel.score_samples(devdata)
dev_gmm_prob = dev_genuine_prob - dev_spoof_prob
_,eer_threholds=CalEER(devlabel,dev_gmm_prob)
# eval predictions
genuine_y_prob = genuineModel.score_samples(testdata)
spoof_y_prob = spoofModel.score_samples(testdata)
gmm_y_prob = genuine_y_prob - spoof_y_prob
_,eer_threholds=CalEER(expected,gmm_y_prob)
# summarize the fit of the model
gmm_predicted = np.where(gmm_y_prob>eer_threholds, 1, 0)
analysisModel(expected,gmm_predicted)  
np.savetxt('./consequence/predictedGMM.txt', gmm_predicted, fmt='%01d')
print("***************************************************************")


def GridSearchTrainModel(model,parameters,scoring_fnc,splits,rate):
    grid = GridSearchCV(model,parameters,scoring=scoring_fnc,cv=StratifiedShuffleSplit(n_splits=splits,test_size=rate,random_state=0))
    grid = grid.fit(traindata,trainlabel)
    CLF = grid.best_estimator_
    print 'Best Parameters found on CV:',grid.best_params_
    return CLF
def DataAnalysis(CLF):
    print '-----devSet-----'
    dev_y_prob = CLF.predict_proba(devdata)
    CalEER(devlabel,dev_y_prob[:,1])
    print '-----evalSet----'
    y_prob = CLF.predict_proba(testdata)
    CalEER(expected,y_prob[:,1])

    predicted = CLF.predict(testdata)
    analysisModel(expected,predicted)
    return predicted
'''
#Logistic Regression:8.74%
print("*********************Logistic Regression***********************")
model = LogisticRegression(n_jobs=1)
#CLF = joblib.load('./ML_model_parameters/LR.pkl')
parameters = {'penalty':('l1','l2'),'C':(0.1,0.3,0.6,1,10,30,60,100),'max_iter':range(50,110,10)}#C=inverse of regularazation
CLF=GridSearchTrainModel(model,parameters,scoring_fnc,10,0.1)
predicted=DataAnalysis(CLF)
np.savetxt('./consequence/predictedLR.txt', predicted, fmt='%01d')
# save model
joblib.dump(CLF,'./ML_model_parameters/LR.pkl')
print("***************************************************************")

#SVM:8.75%
print("*****************************SVM*******************************")
model = SVC(probability=True)
#CLF = joblib.load('./ML_model_parameters/SVM.pkl')
parameters = {'kernel':('linear','rbf'),'C':(0.1,0.3,1,10,30,60,100,1000),'gamma':(1e-2,1e-3,1e-4)}
CLF=GridSearchTrainModel(model,parameters,scoring_fnc,10,0.2)
predicted=DataAnalysis(CLF)
np.savetxt('./consequence/predictedSVM.txt', predicted, fmt='%01d')
# save model
joblib.dump(CLF,'./ML_model_parameters/SVM.pkl')
print("***************************************************************")

#RF:6.12%,8.4%
print("****************************Random Forest**********************")    
model = RandomForestClassifier(random_state=0)
#CLF = joblib.load('./ML_model_parameters/RF.pkl')
parameters = {'n_estimators':range(10,101,10),'max_depth':range(10,21)}
CLF=GridSearchTrainModel(model,parameters,scoring_fnc,5,0.3)
predicted=DataAnalysis(CLF)
np.savetxt('./consequence/predictedRF.txt', predicted, fmt='%01d')
# save model
joblib.dump(CLF,'./ML_model_parameters/RF.pkl')
print("***************************************************************")

#AdaBoost:9.73%
print("***********************AdaBoost********************************")
model = AdaBoostClassifier()
#CLF = joblib.load('./ML_model_parameters/AdaBoost.pkl')
parameters = {'n_estimators':range(10,101,10),'learning_rate':np.arange(0.05,1.1,0.05)}
CLF=GridSearchTrainModel(model,parameters,scoring_fnc,10,0.1)
predicted=DataAnalysis(CLF)
np.savetxt('./consequence/predictedABoost.txt', predicted, fmt='%01d')
# save model
joblib.dump(CLF,'./ML_model_parameters/AdaBoost.pkl')
print("***************************************************************")

#Gradient Boosting Decision Tree:9.05%
#Best Parameters found on CV: {'n_estimators': 70, 'subsample': 0.6, 'learning_rate': 0.3, 'max_depth': 8}
print("*****************************GBDT******************************")
model = GradientBoostingClassifier(max_features='auto')
#CLF = joblib.load('./ML_model_parameters/GBDT.pkl')
parameters = {'learning_rate':np.arange(0.05,0.5,0.05),'n_estimators':range(10,101,10),
              'max_depth':range(4,21,2),'subsample':np.arange(0.05,1,0.05)}
CLF=GridSearchTrainModel(model,parameters,scoring_fnc,10,0.1)
predicted=DataAnalysis(CLF)
np.savetxt('./consequence/predictedGBDT.txt', predicted, fmt='%01d')
# save model
joblib.dump(CLF,'./ML_model_parameters/GBDT.pkl')
print("***************************************************************")

#lightGBM:6.71,9.18%
#Best Parameters found on CV: {'n_estimators': 50, 'num_leaves': 64, 'learning_rate': 0.10000000000000001}
print("**************************lightGBM****************************")
import lightgbm as lgb
#train_data = lgb.Dataset(traindata,label=trainlabel)
model = lgb.LGBMClassifier(n_jobs=2,metric='auc')
#CLF = joblib.load('./ML_model_parameters/LGBM.pkl')
parameters = {'num_leaves':(32,64,128,256,512,1024),'learning_rate':np.arange(0.05,0.5,0.05),
              'n_estimators':range(10,101,10)}
CLF=GridSearchTrainModel(model,parameters,scoring_fnc,10,0.2)
predicted=DataAnalysis(CLF)
np.savetxt('./consequence/predictedLGBM.txt', predicted, fmt='%01d')
# save model
joblib.dump(CLF,'./ML_model_parameters/LGBM.pkl')
print("***************************************************************")

#Decision Tree:9.70%
#Best Parameters found on CV: {'max_depth': 3}
print("*************************Decision Tree*************************")
model = DecisionTreeClassifier(random_state=0)
#CLF = joblib.load('./ML_model_parameters/DT.pkl')
parameters = {'max_depth':range(3,10)}
CLF=GridSearchTrainModel(model,parameters,scoring_fnc,10,0.1)
print(CLF)
predicted=DataAnalysis(CLF)
np.savetxt('./consequence/predictedDT.txt', predicted, fmt='%01d')
# save model
joblib.dump(CLF,'./ML_model_parameters/DT.pkl')
print("***************************************************************")

#KNN:9.12%
#Best Parameters found on CV: {'n_neighbors': 10}
print("*****************************KNN*******************************")
model = KNeighborsClassifier(algorithm='auto',n_jobs=2)#'algorithm':('ball_tree','kd_tree','brute')
#CLF = joblib.load('./ML_model_parameters/KNN.pkl')
parameters = {'n_neighbors':range(1,11)}
CLF=GridSearchTrainModel(model,parameters,scoring_fnc,10,0.1)
print(CLF)
predicted=DataAnalysis(CLF)
np.savetxt('./consequence/predictedKNN.txt', predicted, fmt='%01d')
# save model
joblib.dump(CLF,'./ML_model_parameters/KNN.pkl')
print("***************************************************************")

#dev:7.3% eval:9.03%
def __create_augmented_train(model,unlabled_data,X, y,sample_rate=0.2):
	     totalNum = unlabled_data.shape[0]
	     num_of_samples = int(totalNum * sample_rate)
	     idx = np.random.randint(totalNum,size=num_of_samples)
	     # Train the model and creat the pseudo-labels
	     #model.fit(X, y)
	     pseudo_label = gmm_predicted#model.predict(unlabled_data)
	     pseudo_data = unlabled_data
         
	     sampled_pseudo_data = pseudo_data[idx,:]
	     print y.shape
	     sampled_pseudo_label = pseudo_label[idx];print sampled_pseudo_label.shape
	     augemented_train_data = np.vstack((X,sampled_pseudo_data));print augemented_train_data.shape
	     augemented_train_label = np.hstack((y,sampled_pseudo_label));print augemented_train_label.shape
	     augemented_train = np.hstack((augemented_train_data,augemented_train_label.reshape(-1,1)))
	     return shuffle(augemented_train)
peusdo_data=__create_augmented_train(CLF,testdata,traindata,trainlabel)

model = RandomForestClassifier(random_state=0)
parameters = {'n_estimators':range(10,101,10),'max_depth':range(10,21)}
grid = GridSearchCV(model,parameters,scoring=scoring_fnc,cv=StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=0))
grid = grid.fit(peusdo_data[:,:-1],peusdo_data[:,-1])
CLF = grid.best_estimator_
print 'Best Parameters found on CV:',grid.best_params_
predicted=DataAnalysis(CLF)
'''
