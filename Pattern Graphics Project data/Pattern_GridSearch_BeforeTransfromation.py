import pandas as pd
import numpy as np
from matplotlib import style
style.use("ggplot")
from time import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

kf=StratifiedKFold(n_splits=5)

# -------Reading the datasets of the Experiment ---------
DataSet_1=pd.read_csv('Exported_Dataset_Ses_1/All_DS_Ses_1.csv')
DataSet_2=pd.read_csv('Exported_Dataset_Ses_2/All_DS_Ses_2.csv')
DataSet_3=pd.read_csv('Exported_Dataset_Ses_3/All_DS_Ses_3.csv')
#concatinating the datasets
DataSet_All=[DataSet_1,DataSet_2,DataSet_3]
DataSet_Patt_Symm=pd.concat(DataSet_All, axis=0)
DataSet_Patt_Symm.to_csv('DataSets_TT_Patterns/DataSet_Patt_Symm.csv', index=False)

#......Cleaning and Removing missing values of the data
#......Filling the cells which cointaing nan values  by zero in the data set

DataSet_Patt_Symm.to_csv('DataSets_TT_Patterns/DataSet_Patt_Symm.csv',index=False)  

#The dataset separating into label/dependent variabls and Features/independent variables
#separating label/dependent and Feature/independent variables
y=DataSet_Patt_Symm.Testers                                           #.... Labels or class are Users
X=DataSet_Patt_Symm.drop(['Testers','Media'], axis=1)            #.... all features i.e 78

#.... split the data in to a training and a test set .......
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



#..... the Pipeline with the classifiers.......
pipe_svc=SVC()


#..... the SVM model parameters.......
param_grid_svc={'kernel':['rbf','poly','sigmoid','linear'],
            'C':[0.001,0.01,0.1,1,10,100],
            'gamma':[0.001,0.01,0.1,1,10,100]}

pipe_rf=RandomForestClassifier()

    #....  the RF grid parameter 
param_grid_rf={'n_estimators':[int(x) for x in np.linspace(start=10, stop=50, num=10)],
                 'max_features':['auto','sqrt'],
                 'max_depth':[int(x) for x in np.linspace(1,10, num=5)],
                 'min_samples_split':[2,5,8],
                 'min_samples_leaf':[1,2,4],
                 'bootstrap':[True, False]}

    #....  the NN grid parameter 

pipe_nn=MLPClassifier()

grid_param_nn= {'solver': ['lbfgs','adam'], 
                'max_iter': [50,100,150], 
                'alpha': 10.0 ** -np.arange(1, 7), 
                'hidden_layer_sizes':np.arange(5, 12), 
                'random_state':[0,1,2,3,4,5,6,7,8,9]}
    #....  the Decision Tree grid parameter 

pipe_dt=DecisionTreeClassifier()
sample_split_range=list(range(1,13))    
grid_param_dt=[{'min_samples_split':sample_split_range[1:],
                'criterion':['gini','entropy'],
                'min_samples_leaf':sample_split_range,
                'max_depth':sample_split_range,
                'presort':[True,False]}]
 

#..... the SVM model.......

Patt_OutputsSVC=open("Result_withoutNormStd\Patt_OutPutNorm_SVM.txt","a+")
grid_search_svc=GridSearchCV(pipe_svc,param_grid_svc,cv=5)

grid_search_svc.fit(X_train,y_train)
print("Best cross validation with Pipline:",grid_search_svc.best_score_)
print("The test score score with Pipline",grid_search_svc.score(X_test,y_test))

pred_svc=grid_search_svc.predict(X_test)
#the classification report used to compute Precision, Recall, and f1-score all at once. they are evualation methods
class_report_svc=classification_report(y_test,pred_svc)
#...... confusion matrix multi dimensional array out put
confusion_svc=confusion_matrix(y_test,pred_svc)

print("The SVC model with Pipline: The test set score is:\n",grid_search_svc.score(X_test,y_test),file=Patt_OutputsSVC)
print("The SVC model: The best cross validation score with Pipline:\n",grid_search_svc.best_score_, file=Patt_OutputsSVC)
print("The SVC model: The best parameters with Pipline:\n",grid_search_svc.best_params_, file=Patt_OutputsSVC)
print("The SVC model: the classification Report:\n",class_report_svc, file=Patt_OutputsSVC) #........,target_names=["UsersName"]
print("The SVC model: the confusion svc metrics:\n{}".format(confusion_svc),file=Patt_OutputsSVC)

#..... the Random forest model.......

Patt_OutputsRF=open("Result_withoutNormStd\Patt_OutPutNorm_RF.txt","a+")

grid_search_rf=GridSearchCV(pipe_rf,param_grid_rf,cv=5)
grid_search_rf.fit(X_train,y_train)
print("The RF Best cross validation with Pipline:",grid_search_rf.best_score_)
print("The RF test set score score with Pipline",grid_search_rf.score(X_test,y_test))

pred_rf=grid_search_rf.predict(X_test)
#the classification report used to compute Precision, Recall, and f1-score all at once. they are evualation methods
class_report_rf=classification_report(y_test,pred_rf)
#...... confusion matrix multi dimensional array out put
confusion_rf=confusion_matrix(y_test,pred_rf)

print("The RF model with Pipline: The test set score is:\n",grid_search_rf.score(X_test,y_test),file=Patt_OutputsRF)
print("The RF model: The best cross validation score with Pipline:\n",grid_search_rf.best_score_, file=Patt_OutputsRF)
print("The RF model: The best parameters with Pipline:\n",grid_search_rf.best_params_, file=Patt_OutputsRF)
print("The RF model: the classification Report:\n",class_report_rf, file=Patt_OutputsRF) #........,target_names=["UsersName"]
print("The RF model: the confusion svc metrics:\n{}".format(confusion_rf),file=Patt_OutputsRF)

#-----------Decision Tree model----------
grid_search_dt=GridSearchCV(pipe_dt,grid_param_dt,cv=5)
grid_search_dt.fit(X_train,y_train)
pred_dt=grid_search_dt.predict(X_test)
print("DT model: the best cross validation accuracy: ",grid_search_dt.best_score_)
print("DT model: the test set score: ",grid_search_dt.score(X_test,y_test))
print("DT model: the best parameters are: ",grid_search_dt.best_params_)

Patt_OutputsDT=open("Result_withoutNormStd\Patt_OutPutNorm_DT.txt","a+")
print("The DT model with Pipline: The test set score is:\n",grid_search_dt.score(X_test,y_test),file=Patt_OutputsDT)
print("The DT model: The best cross validation score with Pipline:\n",grid_search_dt.best_score_, file=Patt_OutputsDT)
print("The DT model: The best parameters with Pipline:\n",grid_search_dt.best_params_, file=Patt_OutputsDT)
print("The DT model: the classification Report:\n",classification_report(y_test,pred_dt), file=Patt_OutputsDT) 
print("The DT model: the confusion svc metrics:\n",confusion_matrix(y_test,pred_dt),file=Patt_OutputsDT)


#-----------Neural network model----------
grid_search_nn=GridSearchCV(pipe_nn,grid_param_nn,cv=5)
grid_search_nn.fit(X_train,y_train)
pred_nn=grid_search_nn.predict(X_test)
print("the best cross validation score: ",grid_search_nn.best_score_)
print("the test set score: ",grid_search_nn.score(X_test,y_test))
print("the accuracy of the true value and predicted values",classification_report(y_test,pred_nn))
print("the best parameters are:\n",grid_search_nn.best_params_) 

Patt_OutputsNN=open("Result_withoutNormStd\Patt_OutPutNorm_NN.txt","a+")
print("The NN model with Pipline: The test set score is:\n",grid_search_nn.score(X_test,y_test),file=Patt_OutputsNN)
print("The NN model: The best cross validation score with Pipline:\n",grid_search_nn.best_score_, file=Patt_OutputsNN)
print("The NN model: The best parameters with Pipline:\n",grid_search_nn.best_params_, file=Patt_OutputsNN)
print("The NN model: the classification Report:\n",classification_report(y_test,pred_nn), file=Patt_OutputsNN) #........,target_names=["UsersName"]
print("The NN model: the confusion svc metrics:\n",confusion_matrix(y_test,pred_nn),file=Patt_OutputsNN)

