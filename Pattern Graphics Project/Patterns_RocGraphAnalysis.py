import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from scipy import interp
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

#np.random.seed(105)
kf=StratifiedKFold(n_splits=5)
## Reading the datasets of all the sessions................
DataSet_1=pd.read_csv('Pattern_Dataset/All_DS_Ses_1.csv')
DataSet_2=pd.read_csv('Pattern_Dataset/All_DS_Ses_2.csv')
DataSet_3=pd.read_csv('Pattern_Dataset/All_DS_Ses_3.csv')
#..... concatinating the two sessions.....
Data_123=[DataSet_1,DataSet_2,DataSet_3]
D123=pd.concat(Data_123, axis=0)


#.... split the data in to a training and a test set .......

y=D123.Testers #.... Labels or class are Users
X=D123.drop(['Testers','Media'], axis=1) #.... all features i.e 78
#---list of users....
list_of_users=['1_User','2_User','3_User','4_User','5_User','6_User','7_User','8_User','9_User','10_User','11_User','12_User','13_User',
               '14_User','15_User','17_User','18_User','19_User','20_User','21_User','22_User','23_User','24_User','25_User','26_User'
               ,'27_User','28_User','29_User','30_User','31_User','32_User']
# binarize the out put(the target or label)
y = label_binarize(y, classes=list_of_users)

n_classes = y.shape[1]
print(X.shape)

#splitting the data to train and test.....
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Learn to predict each class against the other...SVC estimator
classifier_svc = OneVsRestClassifier(SVC(kernel='linear', probability=True))
y_score_svc=classifier_svc.fit(X_train, y_train).predict_proba(X_test)

#---MLPClassifier estimator  ....
classifier_nn = OneVsRestClassifier(MLPClassifier(max_iter=4000, random_state=0))
y_score_nn=classifier_nn.fit(X_train, y_train).predict_proba(X_test)

#---RandomForestClassifier estimator  ....
classifier_rf = OneVsRestClassifier(RandomForestClassifier(min_samples_split=12, random_state=0))
y_score_rf=classifier_rf.fit(X_train, y_train).predict_proba(X_test)

#--- DecisionTreeClassifier estimator  ....
classifier_dt = OneVsRestClassifier(DecisionTreeClassifier(min_samples_leaf=12, random_state=0))
y_score_dt=classifier_dt.fit(X_train, y_train).predict_proba(X_test)


# declare the variables for the Compute ROC curve as dict() and compute compute it
fpr_svc = dict()
tpr_svc = dict()
fnr_svc = dict()
roc_auc_svc= dict()


fpr_nn = dict()
tpr_nn = dict()
fnr_nn = dict()
roc_auc_nn = dict()

fpr_rf = dict()
tpr_rf = dict()
fnr_rf = dict()
roc_auc_rf = dict()

fpr_dt = dict()
tpr_dt = dict()
fnr_dt = dict()
roc_auc_dt = dict()


#....computing the ROC and AUC ....
for i in range(n_classes):
    fpr_svc[i], tpr_svc[i],threshold_svc= roc_curve(y_test[:, i], y_score_svc[:, i])
    roc_auc_svc[i] = auc(fpr_svc[i], tpr_svc[i])# area for each tester using SVC classifier
    fnr_svc[i]=(1-tpr_svc[i])#False Negative Rate for each tester using SVC classifier
    
    fpr_nn[i], tpr_nn[i], _ = roc_curve(y_test[:, i], y_score_nn[:, i])
    roc_auc_nn[i] = auc(fpr_nn[i], tpr_nn[i])# area for each tester using NN classifier
    fnr_nn[i]=(1-tpr_nn[i])#False Negative Rate for each tester using NN classifier
    
    fpr_rf[i],tpr_rf[i], threshold=roc_curve(y_test[:,i],y_score_rf[:,i])
    roc_auc_rf[i]=auc(fpr_rf[i],tpr_rf[i])# area for each tester using Random Forest classifier
    fnr_rf[i]=(1-tpr_rf[i])#False Negative Rate for each tester using Random Forest classifier
    
    fpr_dt[i],tpr_dt[i], threshold=roc_curve(y_test[:,i],y_score_dt[:,i])
    roc_auc_dt[i]=auc(fpr_dt[i],tpr_dt[i])# area for each tester using Decision Tree classifier
    fnr_dt[i]=(1-tpr_dt[i])#False Negative Rate for each tester using Decision Tree classifier


#....computing the EER ....
col_eer=['Tester/class','min_index','fpr[minIndex]','fnr[minIndex]','minAbsValue','EER']
df_err_svc=pd.DataFrame(columns=col_eer)
df_err_nn=pd.DataFrame(columns=col_eer)
df_err_rf=pd.DataFrame(columns=col_eer)
df_err_dt=pd.DataFrame(columns=col_eer)
for j in range(n_classes):
    #....SVC estimator errors
    fpr_svc_eer=[]
    fnr_svc_eer=[]
    min_ds_svc=[]
    EER_svc=[]
    for i in range(len(fpr_svc[j])):
        min_ds_svc.append(np.abs(fpr_svc[j][i]-fnr_svc[j][i]))
        fpr_svc_eer.append(fpr_svc[j][i])
        fnr_svc_eer.append(fnr_svc[j][i])
    min_index=min_ds_svc.index(np.min(min_ds_svc))
    minAbsValue=np.abs(fpr_svc_eer[min_index]-fnr_svc_eer[min_index])
    EER_svc=(fpr_svc_eer[min_index]+fnr_svc_eer[min_index])/2
    df_svc={'Tester/class':list_of_users[j],'min_index':min_index,'fpr[minIndex]':fpr_svc_eer[min_index],'fnr[minIndex]':fnr_svc_eer[min_index],
        'minAbsValue':minAbsValue,'EER':EER_svc}
    df_err_svc=df_err_svc.append(df_svc,ignore_index=True,verify_integrity=False)
    #....NN estimator errors
    fpr_nn_eer=[]
    fnr_nn_eer=[]
    min_ds_nn=[]
    EER_nn=[]
    for i in range(len(fpr_nn[j])):
        min_ds_nn.append(np.abs(fpr_nn[j][i]-fnr_nn[j][i]))
        fpr_nn_eer.append(fpr_nn[j][i])
        fnr_nn_eer.append(fnr_nn[j][i])
    min_index_nn=min_ds_nn.index(np.min(min_ds_nn))
    minAbsValue=np.abs(fpr_nn_eer[min_index_nn]-fnr_nn_eer[min_index_nn])
    EER_nn=(fpr_nn_eer[min_index_nn]+fnr_nn_eer[min_index_nn])/2
    df_nn={'Tester/class':list_of_users[j],'min_index':min_index_nn,
            'fpr[minIndex]':fpr_nn_eer[min_index_nn],'fnr[minIndex]':fnr_nn_eer[min_index_nn],
        'minAbsValue':minAbsValue,'EER':EER_nn}
    df_err_nn=df_err_nn.append(df_nn,ignore_index=True,verify_integrity=False)
    #....RF estimator errors
    fpr_rf_eer=[]
    fnr_rf_eer=[]
    min_ds_rf=[]
    EER_rf=[]
    for i in range(len(fpr_rf[j])):
        min_ds_rf.append(np.abs(fpr_rf[j][i]-fnr_rf[j][i]))
        fpr_rf_eer.append(fpr_rf[j][i])
        fnr_rf_eer.append(fnr_rf[j][i])
    min_index_rf=min_ds_rf.index(np.min(min_ds_rf))
    minAbsValue=np.abs(fpr_rf_eer[min_index_rf]-fnr_rf_eer[min_index_rf])
    EER_rf=(fpr_rf_eer[min_index_rf]+fnr_rf_eer[min_index_rf])/2
    df_rf={'Tester/class':list_of_users[j],'min_index':min_index_rf,
            'fpr[minIndex]':fpr_rf_eer[min_index_rf],'fnr[minIndex]':fnr_rf_eer[min_index_rf],
        'minAbsValue':minAbsValue,'EER':EER_rf}
    df_err_rf=df_err_rf.append(df_rf,ignore_index=True,verify_integrity=False)
    #....DT estimator errors
    fpr_dt_eer=[]
    fnr_dt_eer=[]
    min_ds_dt=[]
    EER_dt=[]
    for i in range(len(fpr_dt[j])):
        min_ds_dt.append(np.abs(fpr_dt[j][i]-fnr_dt[j][i]))
        fpr_dt_eer.append(fpr_dt[j][i])
        fnr_dt_eer.append(fnr_dt[j][i])
    min_index_dt=min_ds_dt.index(np.min(min_ds_dt))
    minAbsValue=np.abs(fpr_dt_eer[min_index_dt]-fnr_dt_eer[min_index_dt])
    EER_dt=(fpr_dt_eer[min_index_dt]+fnr_dt_eer[min_index_dt])/2
    df_dt={'Tester/class':list_of_users[j],'min_index':min_index_dt,
            'fpr[minIndex]':fpr_dt_eer[min_index_dt],'fnr[minIndex]':fnr_dt_eer[min_index_dt],
        'minAbsValue':minAbsValue,'EER':EER_dt}
    df_err_dt=df_err_dt.append(df_dt,ignore_index=True,verify_integrity=False)
#..... Computing global average EER for each estimators....
EER_Aver_svc=np.mean(df_err_svc.loc[:,'EER'])
EER_Aver_nn=np.mean(df_err_nn.loc[:,'EER'])
EER_Aver_rf=np.mean(df_err_rf.loc[:,'EER'])
EER_Aver_dt=np.mean(df_err_dt.loc[:,'EER'])
print(EER_Aver_svc)
errAver=['Classifiers','AverEER']
overAllerrEstimator=pd.DataFrame(columns=errAver)
count=[EER_Aver_svc,EER_Aver_nn,EER_Aver_rf,EER_Aver_dt]
estimator=['SVC','Newural Network','Random Forest',
           'DecisionTreeClassifier'
           ]
for i in range(len(estimator)):
    temp_errAver={'Classifiers':estimator[i],'AverEER':count[i]}
    overAllerrEstimator=overAllerrEstimator.append(temp_errAver,ignore_index=True,verify_integrity=False)

overAllerrEstimator.to_csv('EEROutPuts/overAllerrEstimator.csv')
df_err_svc.to_csv('EEROutPuts/df_err_svc.csv')
df_err_nn.to_csv('EEROutPuts/df_err_nn.csv')
df_err_rf.to_csv('EEROutPuts/df_err_rf.csv')
df_err_dt.to_csv('EEROutPuts/df_err_dt.csv')

#...find the index of the fpr & fnr which gives gives the minimum distance between them
#min_index=min_ds_svc.index(np.min(min_ds_svc))
#minAbsValue=np.abs(fpr_svc_eer[min_index]-fnr_svc_eer[min_index])
#EER=(fpr_svc_eer[min_index]+fnr_svc_eer[min_index])/2

# plotting the DET curves for all estimators...
plt.figure()
plt.plot(fpr_dt[0],fnr_dt[0],color='green',label='DT DET curve ',linestyle='--',lw=3)
plt.plot(fpr_rf[0],fnr_rf[0],color='purple',label='RF DET curve ',linestyle='--',lw=3)
plt.plot(fpr_nn[0],fnr_nn[0],color='orange',label='NN DET curve ',linestyle='--',lw=3)
plt.plot(fpr_svc[0],fnr_svc[0],color='navy',label='SVC DET curve ',linestyle='--',lw=3)
plt.xlabel('False positive rate')
plt.ylabel('False negative rate')
plt.legend(loc='upper right')
plt.title('DET curve using all 4 classifiers for Tester 1 of Pattern Graphics')
plt.savefig('Roc_Graph_Results_Patterns/DET_curve_Tester1.png')


# plotting the DET curves for all estimators tester 20...
plt.figure()
plt.plot(fpr_dt[20],fnr_dt[20],color='green',label='DT DET curve ',linestyle='--',lw=3)
plt.plot(fpr_rf[20],fnr_rf[20],color='purple',label='RF DET curve ',linestyle='--',lw=3)
plt.plot(fpr_nn[20],fnr_nn[20],color='orange',label='NN DET curve ',linestyle='--',lw=3)
plt.plot(fpr_svc[20],fnr_svc[20],color='navy',label='SVC DET curve ',linestyle='--',lw=3)
plt.xlabel('False positive rate')
plt.ylabel('False negative rate')
plt.legend(loc='upper right')
plt.title('DET curve using all 4 classifiers for Tester 20 of Pattern Graphics')
plt.savefig('Roc_Graph_Results_Patterns/DET_curveTester20.png')

# plotting the DET curves for all estimators tester 30...
plt.figure()
plt.plot(fpr_dt[30],fnr_dt[30],color='green',label='DT DET curve ',linestyle='--',lw=3)
plt.plot(fpr_rf[30],fnr_rf[30],color='purple',label='RF DET curve ',linestyle='--',lw=3)
plt.plot(fpr_nn[30],fnr_nn[30],color='orange',label='NN DET curve ',linestyle='--',lw=3)
plt.plot(fpr_svc[30],fnr_svc[30],color='navy',label='SVC DET curve ',linestyle='--',lw=3)
plt.xlabel('False positive rate')
plt.ylabel('False negative rate')
plt.legend(loc='upper right')
plt.title('DET curve using all 4 classifiers for Tester 30 of Pattern Graphics')
plt.savefig('Roc_Graph_Results_Patterns/DET_curveTester30.png')



#....draw the EER for tester tester 1
lw=4
#plt.figure(figsize=(5,5))
plt.figure()
#for i, color in zip(range(n_classes), colors1):
plt.plot(np.sort(fpr_svc[0])[::-1],color='gold',label='FPR SVC',lw=lw)
plt.plot(np.sort(fnr_svc[0]),color='c',label=' FNR SVC',lw=lw)
plt.legend(loc='lower right')
plt.title('EER value at the intersection of FPR vs FRR for Tester 1')
plt.savefig('Roc_Graph_Results_Patterns/EER_fpr_frr.png')

# Compute micro-average ROC curve and ROC area
fpr_svc["micro"], tpr_svc["micro"], _ = roc_curve(y_test.ravel(), y_score_svc.ravel())
roc_auc_svc["micro"] = auc(fpr_svc["micro"], tpr_svc["micro"])


fpr_nn["micro"], tpr_nn["micro"], _ = roc_curve(y_test.ravel(), y_score_nn.ravel())
roc_auc_nn["micro"] = auc(fpr_nn["micro"], tpr_nn["micro"])

fpr_rf["micro"], tpr_rf["micro"], _ = roc_curve(y_test.ravel(), y_score_rf.ravel())
roc_auc_rf["micro"] = auc(fpr_rf["micro"], tpr_rf["micro"])

fpr_dt["micro"], tpr_dt["micro"], _ = roc_curve(y_test.ravel(), y_score_dt.ravel())
roc_auc_dt["micro"] = auc(fpr_dt["micro"], tpr_dt["micro"])
#y_score_ldisc.sort()

# .... using the decison function.....
plt.figure()
lw = 3
plt.plot( fpr_svc[20],tpr_svc[20], color='navy',
         lw=lw, label='ROC curve with SVC(area = %0.2f)' % roc_auc_svc[20])

plt.plot( threshold_svc, color='darkorange',
         lw=lw, label='ROC curve with SVC')
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 2.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Threshold')
plt.ylabel('False Positive Rate')
plt.title('SVC Receiver operating characteristic for tester 20')
plt.legend(loc="upper right")
#plt.show()
plt.savefig('Roc_Graph_Results_Patterns/SVC_Roc20_threshold.png')    
#.... plot the roc for tester 1 using all models applied above....
plt.figure()
lw = 3
plt.plot(fpr_svc[1], tpr_svc[1], color='lime',
         lw=lw, linestyle='-',label='ROC with SVC(AUC area = %0.2f)' % roc_auc_svc[1])
plt.plot(fpr_nn[1], tpr_nn[1], color='darkorange',
         lw=lw, label='ROC with NN(AUC area = %0.2f)' % roc_auc_nn[1])
plt.plot(fpr_rf[1], tpr_rf[1], color='blue',
         lw=lw, linestyle='-',label='ROC with RF(AUC area = %0.2f)' % roc_auc_rf[1])
plt.plot(fpr_dt[1], tpr_dt[1], color='orange',
         lw=lw, linestyle='-',label='ROC with DT(AUC area = %0.2f)' % roc_auc_dt[1])
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for Tester 1 using all classifiers')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/allClassifier_RocTester1.png')    
plt.show()

#.... plot the roc for tester 20 using all models applied above....
plt.figure()
lw = 3
plt.plot(fpr_svc[20], tpr_svc[20], color='lime',
         lw=lw, linestyle='-',label='ROC with SVC(AUC area = %0.2f)' % roc_auc_svc[20])
plt.plot(fpr_nn[20], tpr_nn[20], color='darkorange',
         lw=lw, linestyle='-',label='ROC with NN(AUC area = %0.2f)' % roc_auc_nn[20])
plt.plot(fpr_rf[20], tpr_rf[20], color='blue',
         lw=lw, linestyle='-',label='ROC with RF(AUC area = %0.2f)' % roc_auc_rf[20])
plt.plot(fpr_dt[20], tpr_dt[20], color='orange',
         lw=lw, linestyle='-',label='ROC with DT(AUC area = %0.2f)' % roc_auc_dt[20])
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='-')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for Tester 20 using all classifiers')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/allClassifier_RocTester20.png')    
plt.show()
#.... plot the roc for tester 30 using all models applied above....
plt.figure()
lw = 3
plt.plot(fpr_svc[30], tpr_svc[30], color='lime',
         lw=lw, linestyle='-',label='ROC with SVC(AUC area = %0.2f)' % roc_auc_svc[30])
plt.plot(fpr_nn[30], tpr_nn[30], color='darkorange',
         lw=lw, linestyle='-',label='ROC with NN(AUC area = %0.2f)' % roc_auc_nn[30])
plt.plot(fpr_rf[30], tpr_rf[30], color='blue',
         lw=lw, linestyle='-',label='ROC with RF(AUC area = %0.2f)' % roc_auc_rf[30])
plt.plot(fpr_dt[30], tpr_dt[30], color='orange',
         lw=lw, linestyle='-',label='ROC with DT(AUC area = %0.2f)' % roc_auc_dt[30])
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for Tester 30 using all classifiers')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/allClassifier_RocTester30.png')    
plt.show()

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr_svc = np.unique(np.concatenate([fpr_svc[i] for i in range(n_classes)]))
all_fpr_nn = np.unique(np.concatenate([fpr_nn[i] for i in range(n_classes)]))
all_fpr_rf = np.unique(np.concatenate([fpr_rf[i] for i in range(n_classes)]))
all_fpr_dt = np.unique(np.concatenate([fpr_dt[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr_svc = np.zeros_like(all_fpr_svc)
for i in range(n_classes):
    mean_tpr_svc += interp(all_fpr_svc, fpr_svc[i], tpr_svc[i])

mean_tpr_nn = np.zeros_like(all_fpr_nn)
for i in range(n_classes):
    mean_tpr_nn += interp(all_fpr_nn, fpr_nn[i], tpr_nn[i])

mean_tpr_rf = np.zeros_like(all_fpr_rf)
for i in range(n_classes):
    mean_tpr_rf += interp(all_fpr_rf, fpr_rf[i], tpr_rf[i])
mean_tpr_dt = np.zeros_like(all_fpr_dt)
for i in range(n_classes):
    mean_tpr_dt += interp(all_fpr_dt, fpr_dt[i], tpr_dt[i])

# Finally average it and compute AUC

mean_tpr_svc /= n_classes

fpr_svc["macro"] = all_fpr_svc
tpr_svc["macro"] = mean_tpr_svc
roc_auc_svc["macro"] = auc(fpr_svc["macro"], tpr_svc["macro"])


mean_tpr_nn /= n_classes

fpr_nn["macro"] = all_fpr_nn
tpr_nn["macro"] = mean_tpr_nn
roc_auc_nn["macro"] = auc(fpr_nn["macro"], tpr_nn["macro"])

mean_tpr_rf /= n_classes

fpr_rf["macro"] = all_fpr_rf
tpr_rf["macro"] = mean_tpr_rf
roc_auc_rf["macro"] = auc(fpr_rf["macro"], tpr_rf["macro"])

mean_tpr_dt /= n_classes

fpr_dt["macro"] = all_fpr_dt
tpr_dt["macro"] = mean_tpr_dt
roc_auc_dt["macro"] = auc(fpr_dt["macro"], tpr_dt["macro"])


# Plot all micro averaging ROC curves for all kinds of classifier
plt.figure()
plt.plot(fpr_svc["micro"], tpr_svc["micro"],
         label='micro-Aver ROC curve SVC(area = {0:0.2f})'
               ''.format(roc_auc_svc["micro"]),
         color='navy', linewidth=3)
plt.plot(fpr_nn["micro"], tpr_nn["micro"],
         label='micro-Aver ROC curve NN (area = {0:0.2f})'
               ''.format(roc_auc_nn["micro"]),
         color='lime', linewidth=3)
plt.plot(fpr_rf["micro"], tpr_rf["micro"],
         label='micro-Aver ROC curve RandomForest (area = {0:0.2f})'
               ''.format(roc_auc_rf["micro"]),
         color='orange', linewidth=3)
plt.plot(fpr_dt["micro"], tpr_dt["micro"],
         label='micro-Aver ROC curve DecisionTree(area = {0:0.2f})'
               ''.format(roc_auc_dt["micro"]),
         color='green', linewidth=3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Micro Averaging Roc curve for patterns with 4 classifiers')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/MicroAver_Pattern_curves.png')


# Plot all macro averaging ROC curves for all kinds of classifier
plt.figure()
plt.plot(fpr_svc["macro"], tpr_svc["macro"],
         label='macro-Aver ROC curve SVC(area = {0:0.2f})'
               ''.format(roc_auc_svc["macro"]),
         color='navy', linewidth=3)
plt.plot(fpr_nn["macro"], tpr_nn["macro"],
         label='macro-Aver ROC curve NN (area = {0:0.2f})'
               ''.format(roc_auc_nn["macro"]),
         color='lime', linewidth=3)
plt.plot(fpr_rf["macro"], tpr_rf["macro"],
         label='macro-Aver ROC curve RandomForest (area = {0:0.2f})'
               ''.format(roc_auc_rf["macro"]),
         color='orange',  linewidth=3)
plt.plot(fpr_dt["macro"], tpr_dt["macro"],
         label='macro-Aver ROC curve DecisionTree(area = {0:0.2f})'
               ''.format(roc_auc_dt["macro"]),
         color='green', linewidth=3)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Macro Averaging Roc curve for Patterns with 4 classifiers')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/MacroAver_Patterns_curves.png')

colors = cycle(['aqua', 'darkorange', 'cornflowerblue','r','g','b','c','m','y','orange','brown',
                'gray','purple','gold','olive','lime','peru','yellow','blue','navy'])
#plt.figure(figsize=(5,5))
plt.figure()
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_svc[i], tpr_svc[i], color=color, lw=lw,
             label='with SVC ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc_svc[i]))
    
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class Support Vector C (for all testers)')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/all_testers_roc_PatternSVC.png')
plt.show()

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_nn[i], tpr_nn[i], color=color, lw=lw,
             label='with NN ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc_nn[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class Random Forest (for all testers)')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/all_testers_roc_patternNN.png')
plt.show()
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_rf[i], tpr_rf[i], color=color, lw=lw,
             label='with RF ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc_rf[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class Random Forest (for all testers)')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/all_testers_roc_PatternRF.png')
plt.show()

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr_dt[i], tpr_dt[i], color=color, lw=lw,
             label='with DT ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc_dt[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class Decision Tree(for all testers)')
plt.legend(loc="lower right")
plt.savefig('Roc_Graph_Results_Patterns/all_testers_roc_PatternDT.png')
plt.show()

