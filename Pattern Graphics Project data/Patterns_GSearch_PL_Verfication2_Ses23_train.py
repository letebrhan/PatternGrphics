import pandas as pd
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
#np.random.seed(105)
kf=StratifiedKFold(n_splits=5)
## Reading the datasets of all the sessions................
DataSet_1=pd.read_csv('Pattern_Dataset/All_DS_Ses_1.csv')
DataSet_2=pd.read_csv('Pattern_Dataset/All_DS_Ses_2.csv')
DataSet_3=pd.read_csv('Pattern_Dataset/All_DS_Ses_3.csv')
#..... concatinating the two sessions
Data_12=[DataSet_2,DataSet_3]
D12=pd.concat(Data_12, axis=0)

#...... session one used for testing
D3=DataSet_1

#..... the Pipeline with the classifiers.......#..... the SVM model parameters.......

param_grid_svc={'svm__kernel':['rbf','poly','sigmoid','linear'],
            'svm__C':[0.001,0.01,0.1,1,10,100],
            'svm__gamma':[0.001,0.01,0.1,1,10,100]}

temp_aver=[]
col_aver=['totalTime','globalAccuracy','globalSensiti','globalSpecifi','gloFRR','gloFAR','gloTAR','gloERR','gloF_score']
tempAver_PatternM2=pd.DataFrame(columns=col_aver)

list_of_users=['1_User','2_User','3_User','4_User','5_User','6_User','7_User','8_User','9_User','10_User','11_User','12_User','13_User',
               '14_User','15_User','17_User','18_User','19_User','20_User','21_User','22_User','23_User','24_User','25_User','26_User'
               ,'27_User','28_User','29_User','30_User','31_User','32_User']


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
    
    time0=time()
    #----iterating for each users----
    for t in range(len(list_of_users)):
        #--- Selecting feature vectors of tester T from D12 dataset-------
        SingleTester12=D12.Testers==list_of_users[t] # this array contains the list of users listed above
        SingleTesterData12=D12[SingleTester12]
        #.... assigning label= 1 to the tester T(SingleTesterData12)
        SingleTesterData12.Testers=1
        #--- data set of the rest testers with thier feature vectors  (exculding tester T)     
        RestData12=D12.Testers!=list_of_users[t]
        RestTesterData12=D12[RestData12]   
        #------- Select the feature vectors of tester T from D3-----
        SingleTester3=D3.Testers==list_of_users[t]
        SingleTesterData3=D3[SingleTester3]
        #--- assign label=1 tester T selected from D3
        SingleTesterData3.Testers=1
        #------- data set of the rest testers in D3 with thier feature vectors       
        RestData3=D3.Testers!=list_of_users[t]
        RestTesterData3=D3[RestData3]
        #....repeating the random selection and the prediction for iter_no number of times
        iter_no=30# the number of iterations
        TP=0
        FP=0
        FN=0
        TN=0
        EER=0
        Accuracy=[]
        classAccuracy=0
        mean2error=0
        for itr in range(iter_no):   
            #--- randomly select the same number of feature vectors from 'RestTesterData12' which are not cointained in tester T(SingleTesterData12)
            random_subset12 = RestTesterData12.sample(n=20)
            #.... assigning label 0 to the other 20 randomly selected feature vector of the testers
            random_subset12.Testers=0
            #concatinate the randomly selected feature vectors and the testet T feature vectors
            Data12=[SingleTesterData12,random_subset12]
            D=pd.concat(Data12,axis=0)        
            # splitting the new Data set D in to train test sets
            y=D.Testers
            X=D.drop(['Testers','Media'],axis=1)
            X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)
            #----Feature selection using SelectKbest------
            kbest_svc=SelectKBest(f_classif,k=c[j])
            kbest_svc.fit(X_train,y_train)
            X_train_kb=kbest_svc.transform(X_train)
            X_test_kb=kbest_svc.transform(X_test)
            pipe_svc_MinMax_kbest= Pipeline([("scaler", preprocessing.MinMaxScaler()), ('kbest_svc', kbest_svc), ("svm", SVC())])
            ##---- For all feature considerations-------
            pipe_svc_MinMax= Pipeline([("scaler", preprocessing.MinMaxScaler()), ("svm", SVC())])
            #-----The Grid search with Stratified KFold cross validation kf=5
            grid_search_svc_kb=GridSearchCV(pipe_svc_MinMax_kbest,param_grid_svc,cv=kf)
            grid_search_svc_kb.fit(X_train_kb,y_train)# training the model on the data set of D           
            #--- randomly select the same number of feature vectors from 'RestTesterData3' which are not cointained in tester T(SingleTesterData3)
            random_subset3=RestTesterData3.sample(n=10)
            #--- assign label 0 to the class
            random_subset3.Testers=0
            #create the new Data set D1 from random_subset3 + SingleTester3       
            Data3=[SingleTesterData3,random_subset3]
            D1=pd.concat(Data3,axis=0)
            #----separating Dataset D1 which contain feature vector of tester T and other 10 features into label and features
            y_testD1=D1.Testers
            X_testD1=D1.drop(['Testers','Media'],axis=1)
            #--- transform the X_testD1 dataset to K values the feature selector KBest
            X_testD1_kb=kbest_svc.transform(X_testD1)
            #for each feature vector in D1      
            T_tester=SingleTesterData3.loc[:,'Testers']
            D_tester=D1.loc[:,'Testers']
            #----separating tester T into label and features
            y_testT=SingleTesterData3.Testers
            X_testT=SingleTesterData3.drop(['Testers','Media'],axis=1)
            #X_testT_rfe=svc_rfe.transform(X_testT)
            X_testT_kb=kbest_svc.transform(X_testT)
            
            #the classification report used to compute Precision, Recall, and f1-score all at once. they are evualation methods
            pred_svc=grid_search_svc_kb.predict(X_testT_kb)
            class_report_svc=classification_report(y_testT,pred_svc)  
            classAccuracy=grid_search_svc_kb.score(X_testT_kb,y_testT)+classAccuracy
            #...... confusion matrix multi dimensional array out put
            confusion_svc=confusion_matrix(y_testT,pred_svc)
            Patt_OutputsSVC=open("Result_Met2_Pattern_Ses23_train\pattern_OutPutNorm_SVM_Kbest_Ver2.txt","a+")
            #print("Testers : ",list_of_users[t],"For iteration itr =",iter_no,"The SVC model: The best parameters :\n",grid_search_svc_kb.best_params_,file=Patt_OutputsSVC)
            #print("Testers : ",list_of_users[t],"For iteration itr =",iter_no,"The SVC model: the classification Report:\n",class_report_svc,file=Patt_OutputsSVC) #........,target_names=["UsersName"]
            #print("Testers : ",list_of_users[t],"For iteration itr =",iter_no,"The SVC model: the confusion svc metrics with RFECV :\n",confusion_svc,file=Patt_OutputsSVC)

            for v in range(len(D1)):
                #If the feature vector of D1 is the feature vector of tester T
                #(since D1 is the combination of featur vector of T and some other 10 feature vectors selected from D3 which are not combined in feature vector of tester T)
                for s in range(len(SingleTesterData3)):
                    if D_tester.values[v]==T_tester.values[s]:
                        if (grid_search_svc_kb.predict(X_testD1_kb)[v])==1:
                            TP = TP + 1
                        else:
                            FN = FN + 1
                    else: # other testers not tester T(other testers in D1)                    
                         if (grid_search_svc_kb.predict(X_testD1_kb)[v])==0:
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

    globalAccuracy=np.mean(df_acc.loc[:,'Accuracy']) 
    print("Averge Accuracy...:", globalAccuracy)
    
    globalSensiti=np.mean(df_perfIndicators.loc[:,'Sensitivity(TPR)']) 
    print("Averge Sensitivity(TPR) :",globalSensiti)
    
    globalSpecifi=np.mean(df_perfIndicators.loc[:,'Specificity(TNR)']) 
    print("Averge Specificity(TNR) :",globalSpecifi)
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
    tempAver_PatternM2=tempAver_PatternM2.append(df_aver,ignore_index=True, verify_integrity=False)
tempAver_PatternM2.to_csv('Result_Met2_Pattern_Ses23_train/tempAver_PatternM2.csv')
df_perfIndicators.to_csv('Result_Met2_Pattern_Ses23_train/df_perfIndicators.csv')
Data=[df_acc,df_perfIndicators]
Data_Pattern_Verfi_Meth2=pd.concat(Data,axis=1)
Data_Pattern_Verfi_Meth2.to_csv('Result_Met2_Pattern_Ses23_train/Data_Pattern_Verfi_Meth2.csv')

classAccuT_method2=classAccuT
#classAccuT_method2.to_csv('Result_Met2_Pattern_Ses23_train/classAccuT_method2.csv')