
import pandas as pd
import numpy as np
from matplotlib import style
style.use("ggplot")
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

## -------Reading the datasets of the experiment 3
DataSet_1=pd.read_csv('Exported_Dataset_Ses_1/All_DS_Ses_1.csv')
DataSet_2=pd.read_csv('Exported_Dataset_Ses_2/All_DS_Ses_2.csv')
DataSet_3=pd.read_csv('Exported_Dataset_Ses_3/All_DS_Ses_3.csv')
#concatinating the datasets
DataSet_All=[DataSet_1,DataSet_2,DataSet_3]
DataSet_Patt_Symm=pd.concat(DataSet_All, axis=0)
DataSet_Patt_Symm.to_csv('DataSets_TT_Patterns/DataSet_Patt_Symm.csv', index=False)

#The Train_Data_1_slides_symm dataset separating into label/dependent variabls and Features/independent variables
#separating label/dependent and Feature/independent variables
y=DataSet_Patt_Symm.Testers                                           #.... Labels or class are Users
X=DataSet_Patt_Symm.drop(['Testers','Media'], axis=1)            #.... all features i.e 78

#..........Grid search with cross validation......................................
#.... split the data in to a training and a test set .......
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#....  variable declaration for the Pipeline class with four Preprocessing to be used with SVM
pipe_svc_MinMax=Pipeline([("scaler",preprocessing.MinMaxScaler()),("svm",SVC())])
pipe_svc_MaxAbs=Pipeline([("scaler",preprocessing.MaxAbsScaler()),("svm",SVC())])
pipe_svc_robust=Pipeline([("scaler",preprocessing.RobustScaler()),("svm",SVC())])
pipe_svc_Normalizer=Pipeline([("scaler",preprocessing.Normalizer()),("svm",SVC())])

#..... the SVM model parameters.......
param_grid_svc={'svm__kernel':['rbf','poly','sigmoid','linear'],
            'svm__C':[0.001,0.01,0.1,1,10,100],
            'svm__gamma':[0.001,0.01,0.1,1,10,100]}
#....  variable declaration for the Pipeline class with four Preprocessing to be used with Random Forest

pipe_rf_MinMax=Pipeline([("scaler",preprocessing.MinMaxScaler()),("rf",RandomForestClassifier())])
pipe_rf_MaxAbs=Pipeline([("Abs scaler",preprocessing.MaxAbsScaler()),("rf",RandomForestClassifier())])
pipe_rf_robust=Pipeline([("Robust",preprocessing.RobustScaler()),("rf",RandomForestClassifier())])
pipe_rf_Normalizer=Pipeline([("Normalizer",preprocessing.Normalizer()),("rf",RandomForestClassifier())])

    #....  the Random Forest Classifier grid parameter 
param_grid_rf={'rf__n_estimators':[int(x) for x in np.linspace(start=10, stop=50, num=10)],
                 'rf__max_features':['auto','sqrt'],
                 'rf__max_depth':[int(x) for x in np.linspace(1,10, num=5)],
                 'rf__min_samples_split':[2,5,8],
                 'rf__min_samples_leaf':[1,2,4],
                 'rf__bootstrap':[True, False]}

#....  variable declaration for the Pipeline class with four Preprocessing to be used with Decision Tree
pipe_dt_MinMax=Pipeline([("scaler",preprocessing.MinMaxScaler()),("dt",DecisionTreeClassifier())])
pipe_dt_MaxAbs=Pipeline([("Abs scaler",preprocessing.MaxAbsScaler()),("dt",DecisionTreeClassifier())])
pipe_dt_robust=Pipeline([("Robust",preprocessing.RobustScaler()),("dt",DecisionTreeClassifier())])
pipe_dt_Normalizer=Pipeline([("Normalizer",preprocessing.Normalizer()),("dt",DecisionTreeClassifier())])
   #....  the Decision Tree Classifier grid parameter 

sample_split_range=list(range(1,13))    
param_grid_dt=[{'dt__min_samples_split':sample_split_range[1:],
                'dt__criterion':['gini','entropy'],
                'dt__min_samples_leaf':sample_split_range,
                'dt__max_depth':sample_split_range,
                'dt__presort':[True,False]}]

 
#....  variable declaration for the Pipeline class with four Preprocessing to be used with Neural Network

pipe_nn_MinMax=Pipeline([("scaler",preprocessing.MinMaxScaler()),("nn",MLPClassifier())])
pipe_nn_MaxAbs=Pipeline([("Abs scaler",preprocessing.MaxAbsScaler()),("nn",MLPClassifier())])
pipe_nn_robust=Pipeline([("Robust",preprocessing.RobustScaler()),("nn",MLPClassifier())])
pipe_nn_Normalizer=Pipeline([("Normalizer",preprocessing.Normalizer()),("nn",MLPClassifier())])

   #....  the Neural Network Classifier grid parameter 

param_grid_nn= {'nn__solver': ['lbfgs','adam'], 
                'nn__max_iter': [50,100,150], 
                'nn__alpha': 10.0 ** -np.arange(1, 7), 
                'nn__hidden_layer_sizes':np.arange(5, 12), 
                'nn__random_state':[0,1,2,3,4,5,6,7,8,9]}

#----------SVM model with GridSearchCV function on the Pipeline class with Min Max scaler..............
gridSearch_svc_minMax=GridSearchCV(pipe_svc_MinMax,param_grid_svc,cv=5)
gridSearch_svc_minMax.fit(X_train,y_train)
svcTestAcc_MinMax=gridSearch_svc_minMax.score(X_test,y_test)
svcCrossValAcc_MinMax=gridSearch_svc_minMax.best_score_

#----------SVM model with GridSearchCV function on the Pipeline class with  Max absolute scaler..............

gridSearch_svc_Maxabs=GridSearchCV(pipe_svc_MaxAbs,param_grid_svc,cv=5)
gridSearch_svc_Maxabs.fit(X_train,y_train)
    svcTestAcc_MaxAbs=gridSearch_svc_Maxabs.score(X_test,y_test)
svcCrossValAcc_MaxAbs=gridSearch_svc_Maxabs.best_score_

#----------SVM model with GridSearchCV function on the Pipeline class with  robust scaler..............

gridSearch_svc_robust=GridSearchCV(pipe_svc_robust,param_grid_svc,cv=5)
gridSearch_svc_robust.fit(X_train,y_train)
svcTestAcc_robust=gridSearch_svc_robust.score(X_test,y_test)
svcCrossValAcc_robust=gridSearch_svc_robust.best_score_
#----------SVM model with GridSearchCV function on the Pipeline class with Normalizer scaler..............

gridSearch_svc_norm=GridSearchCV(pipe_svc_Normalizer,param_grid_svc,cv=5)
gridSearch_svc_norm.fit(X_train,y_train)
svcTestAcc_norm=gridSearch_svc_norm.score(X_test,y_test)
svcCrossValAcc_norm=gridSearch_svc_norm.best_score_

Symm_OutputsSVC=open("PAtternsSymm_MinMaxAbs_Robust_Norm\Symm_OutputsSVC.txt","a+")
print("The SVC model: The test set score using MinMax:\n",gridSearch_svc_minMax.score(X_test,y_test),file=Symm_OutputsSVC)
print("The SVC model: The best cross validation score using MinMax:\n",gridSearch_svc_minMax.best_score_, file=Symm_OutputsSVC)
print("The SVC model: The best parameters with selected using MinMax:\n",gridSearch_svc_minMax.best_params_, file=Symm_OutputsSVC)
print("The SVC model: the classification Report using MinMax:\n",classification_report(y_test,gridSearch_svc_minMax.predict(X_test)), file=Symm_OutputsSVC) #........,target_names=["UsersName"]
print("The SVC model: the confusion svc metrics using MinMax:\n{}".format(confusion_matrix(y_test,gridSearch_svc_minMax.predict(X_test))),file=Symm_OutputsSVC)

print("The SVC model: The test set score using Maxabs:\n",gridSearch_svc_Maxabs.score(X_test,y_test),file=Symm_OutputsSVC)
print("The SVC model: The best cross validation score using Maxabs:\n",gridSearch_svc_Maxabs.best_score_, file=Symm_OutputsSVC)
print("The SVC model: The best parameters with selected using Maxabs:\n",gridSearch_svc_Maxabs.best_params_, file=Symm_OutputsSVC)
print("The SVC model: the classification Report using Maxabs:\n",classification_report(y_test,gridSearch_svc_Maxabs.predict(X_test)), file=Symm_OutputsSVC) #........,target_names=["UsersName"]
print("The SVC model: the confusion svc metrics using Maxabs:\n{}".format(confusion_matrix(y_test,gridSearch_svc_Maxabs.predict(X_test))),file=Symm_OutputsSVC)

print("The SVC model: The test set score using Robust:\n",gridSearch_svc_robust.score(X_test,y_test),file=Symm_OutputsSVC)
print("The SVC model: The best cross validation score using Robust:\n",gridSearch_svc_robust.best_score_, file=Symm_OutputsSVC)
print("The SVC model: The best parameters with selected using Robust:\n",gridSearch_svc_robust.best_params_, file=Symm_OutputsSVC)
print("The SVC model: the classification Report using Robust:\n",classification_report(y_test,gridSearch_svc_robust.predict(X_test)), file=Symm_OutputsSVC) #........,target_names=["UsersName"]
print("The SVC model: the confusion svc metrics using Robust:\n{}".format(confusion_matrix(y_test,gridSearch_svc_robust.predict(X_test))),file=Symm_OutputsSVC)

print("The SVC model: The test set score using Normalizer:\n",gridSearch_svc_norm.score(X_test,y_test),file=Symm_OutputsSVC)
print("The SVC model: The best cross validation score using Normalizer:\n",gridSearch_svc_norm.best_score_, file=Symm_OutputsSVC)
print("The SVC model: The best parameters with selected using Normalizer:\n",gridSearch_svc_norm.best_params_, file=Symm_OutputsSVC)
print("The SVC model: the classification Report using Normalizer:\n",classification_report(y_test,gridSearch_svc_norm.predict(X_test)), file=Symm_OutputsSVC) #........,target_names=["UsersName"]
print("The SVC model: the confusion svc metrics using Normalizer:\n{}".format(confusion_matrix(y_test,gridSearch_svc_norm.predict(X_test))),file=Symm_OutputsSVC)

#-----------Random forest model with GridSearchCV function on the Pipeline class with Min Max scaler..............
gridSearch_rf_minMax=GridSearchCV(pipe_rf_MinMax,param_grid_rf,cv=5)
gridSearch_rf_minMax.fit(X_train,y_train)
rfTestAcc_MinMax=gridSearch_rf_minMax.score(X_test,y_test)
rfCrossValAcc_MinMax=gridSearch_rf_minMax.best_score_
#-----------Random forest model with GridSearchCV function on the Pipeline class with  Max absolute scaler..............

gridSearch_rf_Maxabs=GridSearchCV(pipe_rf_MaxAbs,param_grid_rf,cv=5)
gridSearch_rf_Maxabs.fit(X_train,y_train)
rfTestAcc_MaxAbs=gridSearch_rf_Maxabs.score(X_test,y_test)
rfCrossValAcc_MaxAbs=gridSearch_rf_Maxabs.best_score_
#-----------Random forest model with GridSearchCV function on the Pipeline class with  robust scaler..............

gridSearch_rf_robust=GridSearchCV(pipe_rf_robust,param_grid_rf,cv=5)
gridSearch_rf_robust.fit(X_train,y_train)
rfTestAcc_robust=gridSearch_rf_robust.score(X_test,y_test)
rfCrossValAcc_robust=gridSearch_rf_robust.best_score_
#-----------Random forest model with GridSearchCV function on the Pipeline class with Normalizer scaler..............

gridSearch_rf_norm=GridSearchCV(pipe_rf_Normalizer,param_grid_rf,cv=5)
gridSearch_rf_norm.fit(X_train,y_train)
rfTestAcc_norm=gridSearch_svc_norm.score(X_test,y_test)
rfCrossValAcc_norm=gridSearch_rf_norm.best_score_

Symm_OutputsRF=open("PAtternsSymm_MinMaxAbs_Robust_Norm\Symm_OutputsRF.txt","a+")
print("The RF model: The test set score using MinMax:\n",gridSearch_rf_minMax.score(X_test,y_test),file=Symm_OutputsRF)
print("The RF model: The best cross validation score using MinMax:\n",gridSearch_rf_minMax.best_score_, file=Symm_OutputsRF)
print("The RF model: The best parameters with selected using MinMax:\n",gridSearch_rf_minMax.best_params_, file=Symm_OutputsRF)
print("The RF model: the classification Report using MinMax:\n",classification_report(y_test,gridSearch_rf_minMax.predict(X_test)), file=Symm_OutputsRF) #........,target_names=["UsersName"]
print("The RF model: the confusion svc metrics using MinMax:\n{}".format(confusion_matrix(y_test,gridSearch_rf_minMax.predict(X_test))),file=Symm_OutputsRF)

print("The RF model: The test set score using Maxabs:\n",gridSearch_rf_Maxabs.score(X_test,y_test),file=Symm_OutputsRF)
print("The RF model: The best cross validation score using Maxabs:\n",gridSearch_rf_Maxabs.best_score_, file=Symm_OutputsRF)
print("The RF model: The best parameters with selected using Maxabs:\n",gridSearch_rf_Maxabs.best_params_, file=Symm_OutputsRF)
print("The RF model: the classification Report using Maxabs:\n",classification_report(y_test,gridSearch_rf_Maxabs.predict(X_test)), file=Symm_OutputsRF) #........,target_names=["UsersName"]
print("The RF model: the confusion svc metrics using Maxabs:\n{}".format(confusion_matrix(y_test,gridSearch_rf_Maxabs.predict(X_test))),file=Symm_OutputsRF)

print("The RF model: The test set score using Robust:\n",gridSearch_rf_robust.score(X_test,y_test),file=Symm_OutputsRF)
print("The RF model: The best cross validation score using Robust:\n",gridSearch_rf_robust.best_score_, file=Symm_OutputsRF)
print("The RF model: The best parameters with selected using Robust:\n",gridSearch_rf_robust.best_params_, file=Symm_OutputsRF)
print("The RF model: the classification Report using Robust:\n",classification_report(y_test,gridSearch_rf_robust.predict(X_test)), file=Symm_OutputsRF) #........,target_names=["UsersName"]
print("The RF model: the confusion svc metrics using Robust:\n{}".format(confusion_matrix(y_test,gridSearch_rf_robust.predict(X_test))),file=Symm_OutputsRF)

print("The RF model: The test set score using Normalizer:\n",gridSearch_rf_norm.score(X_test,y_test),file=Symm_OutputsRF)
print("The RF model: The best cross validation score using Robust:\n",gridSearch_rf_norm.best_score_, file=Symm_OutputsRF)
print("The RF model: The best parameters with selected using Robust:\n",gridSearch_rf_norm.best_params_, file=Symm_OutputsRF)
print("The RF model: the classification Report using Robust:\n",classification_report(y_test,gridSearch_rf_norm.predict(X_test)), file=Symm_OutputsRF) #........,target_names=["UsersName"]
print("The RF model: the confusion svc metrics using Robust:\n{}".format(confusion_matrix(y_test,gridSearch_rf_norm.predict(X_test))),file=Symm_OutputsRF)

#-----------Decision Tree model with GridSearchCV function on the Pipeline class with Min Max scaler..............
gridSearch_dt_minMax=GridSearchCV(pipe_dt_MinMax,param_grid_dt,cv=5)
gridSearch_dt_minMax.fit(X_train,y_train)
dtTestAcc_MinMax=gridSearch_dt_minMax.score(X_test,y_test)
dtCrossValAcc_MinMax=gridSearch_dt_minMax.best_score_
#-----------Decision Tree model with GridSearchCV function on the Pipeline class with  Max absolute scaler..............

gridSearch_dt_Maxabs=GridSearchCV(pipe_dt_MaxAbs,param_grid_dt,cv=5)
gridSearch_dt_Maxabs.fit(X_train,y_train)
dtTestAcc_MaxAbs=gridSearch_dt_Maxabs.score(X_test,y_test)
dtCrossValAcc_MaxAbs=gridSearch_dt_Maxabs.best_score_
#-----------Decision Tree model with GridSearchCV function on the Pipeline class with robust scaler..............

gridSearch_dt_robust=GridSearchCV(pipe_dt_robust,param_grid_dt,cv=5)
gridSearch_dt_robust.fit(X_train,y_train)
dtTestAcc_robust=gridSearch_dt_robust.score(X_test,y_test)
dtCrossValAcc_robust=gridSearch_dt_robust.best_score_
#-----------Decision Tree model with GridSearchCV function on the Pipeline class with Normalizer..............

gridSearch_dt_norm=GridSearchCV(pipe_dt_Normalizer,param_grid_dt,cv=5)
gridSearch_dt_norm.fit(X_train,y_train)
dtTestAcc_norm=gridSearch_dt_norm.score(X_test,y_test)
dtCrossValAcc_norm=gridSearch_dt_norm.best_score_
 
Symm_OutputsDT=open("PAtternsSymm_MinMaxAbs_Robust_Norm\Symm_OutputsDT.txt","a+")
print("The DT model: The test set score using MinMax:\n",gridSearch_dt_minMax.score(X_test,y_test),file=Symm_OutputsDT)
print("The DT model: The best cross validation score using MinMax:\n",gridSearch_dt_minMax.best_score_, file=Symm_OutputsDT)
print("The DT model: The best parameters using MinMax:\n",gridSearch_dt_minMax.best_params_, file=Symm_OutputsDT)
print("The DT model: the classification Report using MinMax:\n",classification_report(y_test,gridSearch_dt_minMax.predict(X_test)), file=Symm_OutputsDT) 
print("The DT model: the confusion svc metrics using MinMax:\n",confusion_matrix(y_test,gridSearch_dt_minMax.predict(X_test)),file=Symm_OutputsDT)

print("The DT model: The test set score using MaxAbs:\n",gridSearch_dt_Maxabs.score(X_test,y_test),file=Symm_OutputsDT)
print("The DT model: The best cross validation score using MaxAbs:\n",gridSearch_dt_Maxabs.best_score_, file=Symm_OutputsDT)
print("The DT model: The best parameters using MaxAbs:\n",gridSearch_dt_Maxabs.best_params_, file=Symm_OutputsDT)
print("The DT model: the classification Report using MaxAbs:\n",classification_report(y_test,gridSearch_dt_Maxabs.predict(X_test)), file=Symm_OutputsDT) 
print("The DT model: the confusion svc metrics using MaxAbs:\n",confusion_matrix(y_test,gridSearch_dt_Maxabs.predict(X_test)),file=Symm_OutputsDT)

print("The DT model: The test set score using Robust:\n",gridSearch_dt_robust.score(X_test,y_test),file=Symm_OutputsDT)
print("The DT model: The best cross validation score using Robust:\n",gridSearch_dt_robust.best_score_, file=Symm_OutputsDT)
print("The DT model: The best parameters using Robust:\n",gridSearch_dt_robust.best_params_, file=Symm_OutputsDT)
print("The DT model: the classification Report using Robust:\n",classification_report(y_test,gridSearch_dt_robust.predict(X_test)), file=Symm_OutputsDT) 
print("The DT model: the confusion svc metrics using Robust:\n",confusion_matrix(y_test,gridSearch_dt_robust.predict(X_test)),file=Symm_OutputsDT)

print("The DT model: The test set score using Normalizer:\n",gridSearch_dt_norm.score(X_test,y_test),file=Symm_OutputsDT)
print("The DT model: The best cross validation score using Normalizer:\n",gridSearch_dt_norm.best_score_, file=Symm_OutputsDT)
print("The DT model: The best parameters using Normalizer:\n",gridSearch_dt_norm.best_params_, file=Symm_OutputsDT)
print("The DT model: the classification Report using Normalizer:\n",classification_report(y_test,gridSearch_dt_norm.predict(X_test)), file=Symm_OutputsDT) 
print("The DT model: the confusion svc metrics using Normalizer:\n",confusion_matrix(y_test,gridSearch_dt_norm.predict(X_test)),file=Symm_OutputsDT)

#-----------Neural network model with GridSearchCV function on the Pipeline class with Min Max scaler..............
gridSearch_nn_minMax=GridSearchCV(pipe_nn_MinMax,param_grid_nn,cv=5)
gridSearch_nn_minMax.fit(X_train,y_train)
nnTestAcc_MinMax=gridSearch_nn_minMax.score(X_test,y_test)
nnCrossValAcc_MinMax=gridSearch_nn_minMax.best_score_
#-----------Neural network model with GridSearchCV function on the Pipeline class with  Max absolute scaler..............

gridSearch_nn_Maxabs=GridSearchCV(pipe_nn_MaxAbs,param_grid_nn,cv=5)
gridSearch_nn_Maxabs.fit(X_train,y_train)
nnTestAcc_MaxAbs=gridSearch_nn_Maxabs.score(X_test,y_test)
nnCrossValAcc_MaxAbs=gridSearch_nn_Maxabs.best_score_
#-----------Neural network model with GridSearchCV function on the Pipeline class with robust scaler..............

gridSearch_nn_robust=GridSearchCV(pipe_nn_robust,param_grid_nn,cv=5)
gridSearch_nn_robust.fit(X_train,y_train)
nnTestAcc_robust=gridSearch_nn_robust.score(X_test,y_test)
nnCrossValAcc_robust=gridSearch_nn_robust.best_score_
#-----------Neural network model with GridSearchCV function on the Pipeline class with Normalizer..............

gridSearch_nn_norm=GridSearchCV(pipe_nn_Normalizer,param_grid_nn,cv=5)
gridSearch_nn_norm.fit(X_train,y_train)
nnTestAcc_norm=gridSearch_nn_norm.score(X_test,y_test)
nnCrossValAcc_norm=gridSearch_nn_norm.best_score_

Symm_OutputsNN=open("PAtternsSymm_MinMaxAbs_Robust_Norm\Symm_OutputsNN.txt","a+")
print("The NN model with Pipline: The test set score using MimMax:\n",gridSearch_nn_minMax.score(X_test,y_test),file=Symm_OutputsNN)
print("The NN model: The best cross validation score using MimMax:\n",gridSearch_nn_minMax.best_score_, file=Symm_OutputsNN)
print("The NN model: The best parameters using MimMax:\n",gridSearch_nn_minMax.best_params_, file=Symm_OutputsNN)
print("The NN model: the classification Report using MimMax:\n",classification_report(y_test,gridSearch_nn_minMax.predict(X_test)), file=Symm_OutputsNN) #........,target_names=["UsersName"]
print("The NN model: the confusion svc metrics using MimMax:\n",confusion_matrix(y_test,gridSearch_nn_minMax.predict(X_test)),file=Symm_OutputsNN)

print("The NN model with Pipline: The test set score using MaxAbs:\n",gridSearch_nn_Maxabs.score(X_test,y_test),file=Symm_OutputsNN)
print("The NN model: The best cross validation score using MaxAbs:\n",gridSearch_nn_Maxabs.best_score_, file=Symm_OutputsNN)
print("The NN model: The best parameters using MaxAbs:\n",gridSearch_nn_Maxabs.best_params_, file=Symm_OutputsNN)
print("The NN model: the classification Report using MaxAbs:\n",classification_report(y_test,gridSearch_nn_Maxabs.predict(X_test)), file=Symm_OutputsNN) #........,target_names=["UsersName"]
print("The NN model: the confusion svc metrics using MaxAbs:\n",confusion_matrix(y_test,gridSearch_nn_Maxabs.predict(X_test)),file=Symm_OutputsNN)

print("The NN model with Pipline: The test set score using Robust:\n",gridSearch_nn_robust.score(X_test,y_test),file=Symm_OutputsNN)
print("The NN model: The best cross validation score using Robust:\n",gridSearch_nn_robust.best_score_, file=Symm_OutputsNN)
print("The NN model: The best parameters using Robust:\n",gridSearch_nn_robust.best_params_, file=Symm_OutputsNN)
print("The NN model: the classification Report using Robust:\n",classification_report(y_test,gridSearch_nn_robust.predict(X_test)), file=Symm_OutputsNN) #........,target_names=["UsersName"]
print("The NN model: the confusion svc metrics using Robust:\n",confusion_matrix(y_test,gridSearch_nn_robust.predict(X_test)),file=Symm_OutputsNN)

print("The NN model with Pipline: The test set score using Normalizer:\n",gridSearch_nn_norm.score(X_test,y_test),file=Symm_OutputsNN)
print("The NN model: The best cross validation score using Normalizer:\n",gridSearch_nn_norm.best_score_, file=Symm_OutputsNN)
print("The NN model: The best parameters using Normalizer:\n",gridSearch_nn_norm.best_params_, file=Symm_OutputsNN)
print("The NN model: the classification Report using Normalizer:\n",classification_report(y_test,gridSearch_nn_norm.predict(X_test)), file=Symm_OutputsNN) #........,target_names=["UsersName"]
print("The NN model: the confusion svc metrics using Normalizer:\n",confusion_matrix(y_test,gridSearch_nn_norm.predict(X_test)),file=Symm_OutputsNN)

