
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
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
pipe_svc_MinMax=Pipeline([("scaler",preprocessing.MinMaxScaler()),("svm",SVC())])
#..... the SVM model parameters.......
param_grid_svc={'svm__kernel':['rbf','poly','sigmoid','linear'],
            'svm__C':[0.001,0.01,0.1,1,10,100],
            'svm__gamma':[0.001,0.01,0.1,1,10,100]}



    #....  the RF grid parameter 
param_grid_rf={'rf__n_estimators':[int(x) for x in np.linspace(start=10, stop=50, num=10)],
                 'rf__max_features':['auto','sqrt'],
                 'rf__max_depth':[int(x) for x in np.linspace(1,10, num=5)],
                 'rf__min_samples_split':[2,5,8],
                 'rf__min_samples_leaf':[1,2,4],
                 'rf__bootstrap':[True, False]}

#.... k values of the KBest selector
c=[18,25,30,45,55,70,'all']
#---column names and dataFrame values
perf_indicator=['TP','TN','FP','FN','Accuracy_conf','TestSetAccu','CrossValAccu','ERR','EER','FRR',
                   'FAR','Sensitivity(TPR)',
                   'TAR','Specificity(TNR)','F-score','k values']
#---declaring dataFrames for storing the values

df_perfIndica_svc=pd.DataFrame(columns=perf_indicator)
df_perfIndica_rf=pd.DataFrame(columns=perf_indicator)
df_perfIndica_svc_Ave=pd.DataFrame(columns=perf_indicator)
df_perfIndica_rf_Ave=pd.DataFrame(columns=perf_indicator)

for j in range(len(c)):

    #----Feature selection using SelectKbest with SVC------
    kbest_svc=SelectKBest(f_classif,k=c[j])
    kbest_svc.fit(X_train,y_train)
    X_train_kb=kbest_svc.transform(X_train)
    X_test_kb=kbest_svc.transform(X_test)
    pipe_svc_MinMax_kbest= Pipeline([("scaler", preprocessing.MinMaxScaler()), ('kbest_svc', kbest_svc), ("svm", SVC())])
   
    ##---- For all feature considerations-------
    #-----The Grid search with Stratified KFold cross validation kf=5
    gSearch_svc_minMax_kb=GridSearchCV(pipe_svc_MinMax_kbest,param_grid_svc,cv=kf)
    gSearch_svc_minMax_kb.fit(X_train_kb,y_train)# training the model on the data set of D           
  
    TestAcc_svc=gSearch_svc_minMax_kb.score(X_test_kb,y_test)
    CrossValAccu_svc=gSearch_svc_minMax_kb.best_score_ #---the cross validation accuracy
    y_pred_svc=gSearch_svc_minMax_kb.predict(X_test_kb) #---predictor  
 
    confusion_svc=confusion_matrix(y_test,y_pred_svc)#.... confusion_matrix
    cm_normalized = confusion_svc.astype('float') / confusion_svc.sum(axis=1)[:, np.newaxis] 
    
    FP_svc = confusion_svc.sum(axis=0) - np.diag(confusion_svc)  
    FN_svc = confusion_svc.sum(axis=1) - np.diag(confusion_svc)  
    TP_svc = np.diag(confusion_svc) 
    TN_svc = (len(y_test) - (FP_svc + FN_svc + TP_svc))
    # False positive rate(FAR)
    FPR_svc = FP_svc/(FP_svc+TN_svc)
    TPR_svc=TP_svc/(TP_svc+FN_svc)
    # False negative rate(FRR)
    FNR_svc = FN_svc/(TP_svc+FN_svc)
    #computint the accuracy, F-score and ERR from the confusionmatrix
    Accuracy_svc=((TP_svc + TN_svc) / (TP_svc + FP_svc + FN_svc + TN_svc))
    F_score_svc = 2*TP_svc /(2*TP_svc + FP_svc + FN_svc)
    err_svc=((FP_svc+FN_svc) / (TP_svc + FP_svc + FN_svc + TN_svc))# ERR Error Rejection Rate
    tpr_svc=TP_svc/(TP_svc+FN_svc)    #--- True Positive Rate
    fpr_svc=FP_svc/(FP_svc+TN_svc)    #----- False Positive Rate or False Acceptance Rate

    #---computing the EER
    diff_svc=[]
    fpr_svc=[]
    frr_svc=[]
    FNR_svc=np.sort(FNR_svc)[::-1]
    FPR_svc=np.sort(FPR_svc)
    #print(FPR2[12])
    for i in range(len(FPR_svc)):
        diff_svc.append(np.abs(FPR_svc[i]-FNR_svc[i]))
        fpr_svc.append(FPR_svc[i])
        frr_svc.append(FNR_svc[i])
    min_index=diff_svc.index(np.min(diff_svc))
    smal2=np.abs(fpr_svc[min_index]-frr_svc[min_index])
    EER_svc=(fpr_svc[min_index]+frr_svc[min_index])/2

    perfIndicator={'TP':TP_svc,'TN':TN_svc,'FP':FP_svc,'FN':FN_svc,'Accuracy_conf':Accuracy_svc,'TestSetAccu':TestAcc_svc,
                   'CrossValAccu':CrossValAccu_svc,'ERR':err_svc,'EER':EER_svc,
                   'FRR':(FN_svc/(FN_svc+TP_svc)),'FAR':fpr_svc,'Sensitivity(TPR)':tpr_svc,
                   'TAR':1-(FN_svc/(FN_svc+TP_svc)),'Specificity(TNR)':TN_svc/(TN_svc+FP_svc),'F-score':F_score_svc,'k values':c[j]}
    df_perfIndica_svc=df_perfIndica_svc.append(perfIndicator,ignore_index=True, verify_integrity=False)

    Patt_OutputsSVC=open("Patt_MinMax_KBest\Patt_OutputsSVC.txt","a+")
    print("The SVC model: The test set score using MinMax:\n",gSearch_svc_minMax_kb.score(X_test_kb,y_test),file=Patt_OutputsSVC)
    print("The SVC model: The best cross validation score using MinMax:\n",gSearch_svc_minMax_kb.best_score_, file=Patt_OutputsSVC)
    print("The SVC model: The best parameters with selected using MinMax:\n",gSearch_svc_minMax_kb.best_params_, file=Patt_OutputsSVC)
    print("The SVC model: the classification Report using MinMax:\n",classification_report(y_test,gSearch_svc_minMax_kb.predict(X_test_kb)), file=Patt_OutputsSVC) #........,target_names=["UsersName"]
    print("The SVC model: the confusion svc metrics using MinMax:\n{}".format(confusion_matrix(y_test,gSearch_svc_minMax_kb.predict(X_test_kb))),file=Patt_OutputsSVC)
           
    #----Feature selection using SelectKbest with RF------
    kbest_rf=SelectKBest(f_classif,k=c[j])
    kbest_rf.fit(X_train,y_train)
    X_train_kb=kbest_rf.transform(X_train)
    X_test_kb=kbest_rf.transform(X_test)
    pipe_rf_MinMax_kbest= Pipeline([("scaler", preprocessing.MinMaxScaler()), ('kbest_rf', kbest_rf), ("rf", RandomForestClassifier())])

    #-----The Grid search with Stratified KFold cross validation kf=5
    gSearch_rf_minMax_kb=GridSearchCV(pipe_rf_MinMax_kbest,param_grid_rf,cv=kf)
    gSearch_rf_minMax_kb.fit(X_train_kb,y_train)# training the model on the data set of D           
  
    TestAcc_rf=gSearch_rf_minMax_kb.score(X_test_kb,y_test)
    y_pred_rf=gSearch_rf_minMax_kb.predict(X_test_kb)   

    CrossValAccu_rf=gSearch_rf_minMax_kb.best_score_

    confusion_rf=confusion_matrix(y_test,y_pred_rf)#.... confusion_matrix
    cm_normalized = confusion_rf.astype('float') / confusion_rf.sum(axis=1)[:, np.newaxis] 
    
    FP_rf = confusion_rf.sum(axis=0) - np.diag(confusion_rf)  
    FN_rf = confusion_rf.sum(axis=1) - np.diag(confusion_rf)  
    TP_rf = np.diag(confusion_rf) 
    TN_rf = (len(y_test) - (FP_rf + FN_rf + TP_rf))
    # False positive rate(FAR)
    FPR_rf = FP_rf/(FP_rf+TN_rf)
    TPR_rf=TP_rf/(TP_rf+FN_rf)
    # False negative rate(FRR)
    FNR_rf= FN_rf/(TP_rf+FN_rf)
    #computint the accuracy, F-score and ERR from the confusionmatrix
    Accuracy_rf=((TP_rf + TN_rf) / (TP_rf + FP_rf + FN_rf + TN_rf))
    F_score_rf = 2*TP_rf /(2*TP_rf + FP_rf + FN_rf)
    err_rf=((FP_rf+FN_rf) / (TP_rf + FP_rf + FN_rf + TN_rf))# ERR Error Rejection Rate
    tpr_rf=TP_rf/(TP_rf+FN_rf)    #--- True Positive Rate
    fpr_rf=FP_rf/(FP_rf+TN_rf)    #----- False Positive Rate or False Acceptance Rate

    #---computing the EER
    diff_rf=[]
    fpr_rf=[]
    frr_rf=[]
    FNR_rf=np.sort(FNR_rf)[::-1]
    FPR_rf=np.sort(FPR_rf)
    #....................print(FPR2[12])................
    for i in range(len(FPR_rf)):
        diff_rf.append(np.abs(FPR_rf[i]-FNR_rf[i]))
        fpr_rf.append(FPR_rf[i])
        frr_rf.append(FNR_rf[i])
    smal1=np.min(diff_rf)
    min_index_rf=diff_rf.index(np.min(diff_rf))
    smal2=np.abs(fpr_rf[min_index_rf]-frr_rf[min_index_rf])
    EER_rf=(fpr_rf[min_index_rf]+frr_rf[min_index_rf])/2

    perfIndicator_rf={'TP':TP_rf,'TN':TN_rf,'FP':FP_rf,'FN':FN_rf,'Accuracy_conf':Accuracy_rf,'TestSetAccu':TestAcc_rf,
                   'CrossValAccu':CrossValAccu_rf,'ERR':err_rf,'EER':EER_rf,'FRR':(FN_rf/(FN_rf+TP_rf)),
                   'FAR':fpr_rf,'Sensitivity(TPR)':tpr_rf,
                   'TAR':1-(FN_rf/(FN_rf+TP_rf)),'Specificity(TNR)':TN_rf/(TN_rf+FP_rf),'F-score':F_score_rf,'k values':c[j]}
    df_perfIndica_rf=df_perfIndica_rf.append(perfIndicator_rf,ignore_index=True, verify_integrity=False)

    Patt_OutputsRF=open("Patt_MinMax_KBest\Patt_OutputsRF.txt","a+")
    print("The RF model: The test set score using MinMax:\n",gSearch_rf_minMax_kb.score(X_test_kb,y_test),file=Patt_OutputsRF)
    print("The RF model: The best cross validation score using MinMax:\n",gSearch_rf_minMax_kb.best_score_, file=Patt_OutputsRF)
    print("The RF model: The best parameters with selected using MinMax:\n",gSearch_rf_minMax_kb.best_params_, file=Patt_OutputsRF)
    print("The RF model: the classification Report using MinMax:\n",classification_report(y_test,gSearch_rf_minMax_kb.predict(X_test_kb)), file=Patt_OutputsRF) #........,target_names=["UsersName"]
    print("The RF model: the confusion svc metrics using MinMax:\n{}".format(confusion_matrix(y_test,gSearch_rf_minMax_kb.predict(X_test_kb))),file=Patt_OutputsRF)
# writing the result into csv files.....
df_perfIndica_svc.to_csv('Patt_MinMax_KBest/df_perfIndica_svc.csv')
df_perfIndica_rf.to_csv('Patt_MinMax_KBest/df_perfIndica_rf.csv')
