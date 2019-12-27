import pandas as pd
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
#np.random.seed(105)
kf=StratifiedKFold(n_splits=5)
## Reading the datasets of all the sessions................
DataSet_1=pd.read_csv('Exported_Dataset_Ses_1/All_DS_Ses_1.csv')
DataSet_2=pd.read_csv('Exported_Dataset_Ses_2/All_DS_Ses_2.csv')
DataSet_3=pd.read_csv('Exported_Dataset_Ses_3/All_DS_Ses_3.csv')
#.... concatenating the two sessions....
Data_12=[DataSet_1,DataSet_2]
D12=pd.concat(Data_12, axis=0)

#........ Session 3 for testing 
D3=DataSet_3

#---the best features selected from D3 using orange are 25------


#The D12  is used for trining
y=D12.Testers #.... Labels or class are Users
X=D12.drop(['Testers','Media'], axis=1) #.... all features i.e 78


#.... list of users in array......
list_of_users=['1_User','2_User','3_User','4_User','5_User','6_User','7_User','8_User','9_User','10_User','11_User','12_User','13_User',
               '14_User','15_User','17_User','18_User','19_User','20_User','21_User','22_User','23_User','24_User','25_User','26_User'
               ,'27_User','28_User','29_User','30_User','31_User','32_User']
#..... the SVM model parameters.......

param_grid_svc={'svm__kernel':['rbf','poly','sigmoid','linear'],
            'svm__C':[0.001,0.01,0.1,1,10,100],
            'svm__gamma':[0.001,0.01,0.1,1,10,100]}
temp_aver=[]
col_aver=['totalTime','globalAccuracy','globalSensiti','globalSpecifi','gloFRR','gloFAR','gloTAR','gloERR','gloF_score']
tempAver_PatternM1=pd.DataFrame(columns=col_aver)

#-----creating empty dataframes with thier column names----- 
gloAcc_col=['GlobalAccuracy','AverAcc']
df_gloAcc=pd.DataFrame(columns=gloAcc_col)
acc_col=['Testers','Accuracy']
df_acc=pd.DataFrame(columns=acc_col)
perf_indicator=['TP','TN','FP','FN','ERR','FRR','FAR','EER','Sensitivity(TPR)','TAR','Specificity(TNR)','F-score','k values','Iterations']
df_perfIndicators=pd.DataFrame(columns=perf_indicator)

#.... average classification accuracy on each testers
acc_colT=['Testers','AccuracyT','k values','Iterations']
classAccuT=pd.DataFrame(columns=acc_colT)

#.... k values of the KBest selector
c=[18,25,30,45,55,'all']
for j in range(len(c)):
    X_train,X_test,y_train,y_test=train_test_split(X,y)
    #----Feature selection using SelectKbest------
    kbest_svc=SelectKBest(f_classif,k=c[j])
    kbest_svc.fit(X_train,y_train)
    X_train_kb=kbest_svc.transform(X_train)
    X_test_kb=kbest_svc.transform(X_test)
    
    pipe_svc_MinMax_kbest= Pipeline([("scaler", preprocessing.MinMaxScaler()),('kbest_svc', kbest_svc), ("svm", SVC())])#----('kbest_svc', kbest_svc),
    pipe_svc_MinMax= Pipeline([("scaler", preprocessing.MinMaxScaler()), ("svm", SVC())])
    #---- the Grid search with Stratified KFold cross validation kf=5
    grid_search_svc_kb=GridSearchCV(pipe_svc_MinMax_kbest,param_grid_svc,cv=kf)
    grid_search_svc_kb.fit(X_train_kb,y_train)
   
    #---- For all feature considerations-------
    time0=time()  
        
    #----iterating for each users----
    for t in range(len(list_of_users)):
        #--- the selected tester t data with its feature vectors
        SingleTester=D3.Testers==list_of_users[t]
        SingleTesterData=D3[SingleTester]
        #--- data of the rest tester with thier feature vectors       
        RestData=D3.Testers!=list_of_users[t]
        RestTesterData=D3[RestData]
        #....repeating it for some number of times
        iter_no=30
        TP=0
        FP=0
        FN=0
        TN=0
        EER=0
        Accuracy=[]
        classAccuracy=0
        for itr in range(iter_no):   
            #--- randomly select the same nomber of feature vectors from 'RestTesterData' which are not cointained in tester T
            random_subset = RestTesterData.sample(n=10)
            #concatinate the randomly selected feature vectors and the testet T feature vectors
            Data=[SingleTesterData,random_subset]
            D=pd.concat(Data,axis=0) 
            #----separating Dataset D which contain feature vector of tester T and other 10 features into label and feature
            y_testD=D.Testers
            X_testD=D.drop(['Testers','Media'],axis=1)
            X_testD_kb=kbest_svc.transform(X_testD)
    
            #----separating tester T into label and features
            y_testT=SingleTesterData.Testers
            X_testT=SingleTesterData.drop(['Testers','Media'],axis=1)
            X_testT_kb=kbest_svc.transform(X_testT)
            #the classification report used to compute Precision, Recall, and f1-score all at once. they are evualation methods
            pred_svc=grid_search_svc_kb.predict(X_testT_kb)
            class_report_svc=classification_report(y_testT,pred_svc)  
            classAccuracy=grid_search_svc_kb.score(X_testT_kb,y_testT)+classAccuracy
            #mean2error=grid_search_svc_kb.mean_squared_error(y_testT,pred_svc)+mean2error
            #...... confusion matrix multi dimensional array out put
            confusion_svc=confusion_matrix(y_testT,pred_svc)
            Patt_OutputsSVC=open("Result_Met1_Pattern_Ses12_train\Symm_OutPutNorm_SVM_RFECV_Ver1.txt","a+")
                #print("For iteration i =",i,"The SVC model: The optimal number of features with RFECV :\n",svc_rfe.n_features_,file=Patt_OutputsSVC)
                #print("For iteration i =",i,"The SVC model: The best features selected with RFECV :",X_train.columns[svc_rfe.support_],file=Patt_OutputsSVC)
            print("Testers : ",list_of_users[t],"For iteration itr =",itr,"The SVC model: The best parameters with  :\n",grid_search_svc_kb.best_params_,file=Patt_OutputsSVC)
            print("Testers : ",list_of_users[t],"For iteration itr =",itr,"The SVC model: the classification Report with :\n",class_report_svc,file=Patt_OutputsSVC) #........,target_names=["UsersName"]
            print("Testers : ",list_of_users[t],"For iteration itr =",itr,"The SVC model: the confusion svc metrics with  :\n",confusion_svc,file=Patt_OutputsSVC)    
            #for each feature vector in D
            T_tester=SingleTesterData.loc[:,'Testers']
            D_tester=D.loc[:,'Testers']      
            for v in range(len(D)):
                #If the feature vector of D is the feature vector of tester T
                #(since D is the combination of featur vector of T and some other 10 feature vectors selected from D3 which are not combined in feature vector of tester T)
                for s in range(len(SingleTesterData)):
                    if D_tester.values[v]==T_tester.values[s]:
                        if (grid_search_svc_kb.predict(X_testD_kb)[v])==y_testT.values[s]:
                            TP = TP + 1
                            #print(TP)
                        else:
                            FN = FN + 1
                    else: # other testers not t
                        
                         if (grid_search_svc_kb.predict(X_testD_kb)[v])!=y_testT.values[s]:
                            TN = TN + 1
                         else:
                            FP = FP + 1 
            #print("tester :",list_of_users[t],"TP:",TP)
        TP=TP/iter_no
        TN=TN/iter_no
        FP=FP/iter_no
        FN=FN/iter_no
        Accuracy=((TP + TN) / (TP + FP + FN + TN))
        F_score = 2*TP /(2*TP + FP + FN)
        err=((FP+FN) / (TP + FP + FN + TN))# ERR Error Rejection Rate
        tpr=TP/(TP+FN)    #--- True Positive Rate
        fpr=FP/(FP+TN)    #----- False Positive Rate or False Acceptance Rate
        classAccuracy=classAccuracy/iter_no
        if tpr==fpr:
            EER=EER+1
        accuracy={'Testers':list_of_users[t],'Accuracy':Accuracy}
        df_acc=df_acc.append(accuracy,ignore_index=True, verify_integrity=False)
        perfIndicator={'TP':TP,'TN':TN,'FP':FP,'FN':FN,'ERR':err,'FRR':(FN/(FN+TP)),'FAR':fpr,'EER':EER,'Sensitivity(TPR)':tpr,'TAR':1-(FN/(FN+TP)),'Specificity(TNR)':TN/(TN+FP),'F-score':F_score,'k values':c[j],'Iterations':iter_no}
        df_perfIndicators=df_perfIndicators.append(perfIndicator,ignore_index=True, verify_integrity=False)
        
        classAccuTester={'Testers':list_of_users[t],'AccuracyT':classAccuracy,'k values':c[j],'Iterations':iter_no}
        classAccuT=classAccuT.append(classAccuTester,ignore_index=True, verify_integrity=False)

    Data=[df_acc,df_perfIndicators]
    Data_Accuracy=pd.concat(Data,axis=1)
    usersAcc1=[df_acc,df_perfIndicators]
    Users_AccuracyVer1_rfe=pd.concat(usersAcc1,axis=1)
    #Users_AccuracyVer1_rfe.to_csv('ExportAllSymm Verifi Method1/Users_AccuracySymmVer1_kb.csv')
    
    globalAccuracy=np.mean(df_acc.loc[:,'Accuracy']) 
    #print("Averge Accuracy...:", globalAccuracy)
    
    globalSensiti=np.mean(df_perfIndicators.loc[:,'Sensitivity(TPR)']) 
    #print("Averge Sensitivity(TPR) :",globalSensiti)
    
    globalSpecifi=np.mean(df_perfIndicators.loc[:,'Specificity(TNR)']) 
    #print("Averge Specificity(TNR) :",globalSpecifi)
    gloFRR=np.mean(df_perfIndicators.loc[:,'FRR'])
    gloFAR=np.mean(df_perfIndicators.loc[:,'FAR'])
    gloTAR=np.mean(df_perfIndicators.loc[:,'TAR'])
    gloERR=np.mean(df_perfIndicators.loc[:,'ERR'])
    gloF_score=np.mean(df_perfIndicators.loc[:,'F-score'])
    
    time1=time()
    totalTime=time1-time0
    print("The SVC model: the total time taken to do the analysis with RFECV :\n",totalTime)#,file=Patt_OutputsSVC)
    #..... saving the average values in csv files
    df_aver={'totalTime':totalTime,'globalAccuracy':globalAccuracy,'globalSensiti':globalSensiti,'globalSpecifi':globalSpecifi,
             'gloFRR':gloFRR,'gloFAR':gloFAR,'gloTAR':gloTAR,'gloERR':gloERR,'gloF_score':gloF_score,'k values':c[j]}
    tempAver_PatternM1=tempAver_PatternM1.append(df_aver,ignore_index=True, verify_integrity=False)
tempAver_PatternM1.to_csv('Result_Met1_Pattern_Ses12_train/tempAver_PatternM1.csv')
df_perfIndicators.to_csv('Result_Met1_Pattern_Ses12_train/df_perfIndicators.csv')

Data=[df_acc,df_perfIndicators]
Data_4SideSym_Verfi_Meth1=pd.concat(Data,axis=1)
Data_4SideSym_Verfi_Meth1.to_csv('Result_Met1_Pattern_Ses12_train/Data_4SideSym_Verfi_Meth1.csv')
classAccuT_method1=classAccuT
#classAccuT_method1.to_csv('Result_Met1_Pattern_Ses12_train/classAccuT_method1.csv')
