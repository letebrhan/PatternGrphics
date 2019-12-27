Short cuts: 
	FPR = False Positive Rate
	FNR = False Negative Rate
	EER = Equal Error Rate
	TP = True Positive
	TN = True Negative
	FP = False Positive
	FN = False Negative
	ERR = Error Rejection Rate
	FRR = False Rejecion Rate
	FAR = False Acceptance Rate
	Sensitivity(TPR)= True Positive Rate
	TAR = True Acceptance Rate
	Specificity(TNR) = True Negative Rate	

classifier Short cust:
	SVM = Support Vector Machine
	RF = Random Forest
	DT = Decision Tree
	NN = Neural Network
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

1. the folder "EEROutPuts test size 0.2" contains csv files i.e
	1.1 "df_err_dt.csv" is result computed for each class/tester using Decision Tree Classifier
 	1.2 "df_err_nn.csv" is result computed for each class/tester using Neural Network Classifier
	1.3 "df_err_rf.csv" is result computed for each class/tester  using Random Forest Classifier
 	1.4 "df_err_svc.csv" is result computed for each class/tester using SVM Classifier
 			
	a) all the files from 1.1 to 1.4: Each file contains values of the minimum index, FPR and FNR at minimum index, and EER for each class using each classifier
	b) the file "overAllerrEstimator2.csv" contains the avarage EER for each classifier

 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

2. The "Patt_MinMax_KBest" folder containes:text and csv files.

	2.1 The "Patt_OutputsSVC.txt" contains values of Accuracy, Precision, Recall, and F-score on test set and cross validation accuracy (%) with different values of K Best Feature Selector on Pattern dataset using SVM model
	2.2 The "Patt_OutputsRF.txt" contains values of Accuracy, Precision, Recall, and F-score on test set and cross validation accuracy (%) with different values of K Best Feature Selector on Pattern dataset using Random Forest model
	2.3 The "df_perfIndica_svc.csv" and "df_perfIndica_rf.csv"  csv files whic containes the same values like: Test set Accuracy(TestSetAccu),TP,TN, FP, FN, Cross validation accuracy(CrossValAccu),ERR, EER, FRR, FAR, Sensitivity(TPR), TAR,	Specificity(TNR), and F-score, 	are computed for the k values selected at a time.
	2.4 The Values in our tables in the thesis: The Test set Accuracy, and cross validation accuracy are taken from the csv file
						: The Test set Precision, Test set recall and Fscore are taken from the text file which computed using the confusion matrix.
	
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
3. The "Pattern Data set" folder contain three data sets
	3.1 All_DS_Ses_1.csv is the first data set for the session one experiment
	3.2 All_DS_Ses_2.csv is the second data set for the session two experiment
	3.3 All_DS_Ses_3.csv is the third data set for the session three experiment

 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

4. The "PatternSym_MinMaxAbsRobust_Norm" folder contains the result of the four classifier on the four data tranformations
	4.1 Patt_OutputsDT.txt is the Decision Tree classifer on the MinMax Scaler, Max Absolute Scaler, Robust and Normalizer Data preprocessing
	4.2 Patt_OutputsNN.txt is the Neural Network classifer on the MinMax Scaler, Max Absolute Scaler, Robust and Normalizer Data preprocessing
	4.3 Patt_OutputsRF.txt is the Random Forest classifer on the MinMax Scaler, Max Absolute Scaler, Robust and Normalizer Data preprocessing
	4.4 Patt_OutputsSVC.txt is the Support Vector Machine classifer on the MinMax Scaler, Max Absolute Scaler, Robust and Normalizer Data preprocessing

 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


5. The "Result_Met1_Pattern_Ses12_train" folder containes the result of Verfication method one using session one and two for training and session three for testing
	5.1 Data_Patt_Verfi_Meth1.csv file contains the values of each testers of: Accuracy,	TP,	TN,	FP,	FN,	ERR,	FRR,	FAR,	EER,	Sensitivity(TPR),	TAR,	Specificity(TNR),	
	F-score,	k values,	and the number of Iterations for each tester

 	5.1 tempAver_PatternM1.csv file contains the average result of: global Accuracy,	global Sensitivity,	global Specificity,	global FRR,	global FAR,	global TAR,	global ERR, 	
	global F_score and the	k values with total time taken for each K value features selected for overall testers.
 
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
6. The  "Result_Met2_Pattern_Ses12_train" folder containes the result of Verfication method two using session one and two for training and session three for testing
	6.1 Data_Pattern_Verfi_Meth2.csv file contains the values of each testers of: Accuracy,	TP,	TN,	FP,	FN,	ERR,	FRR,	FAR,	EER,	Sensitivity(TPR),	TAR,	Specificity(TNR),	
	F-score,	k values,	and the number of Iterations for each tester

 	6.1 tempAver_PatternM2.csv file contains the average result of: global Accuracy,	global Sensitivity,	global Specificity,	global FRR,	global FAR,	global TAR,	global ERR, 	
	global F_score and the	k values with total time taken for each K value features selected for overall testers.
 
 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
7. The  "Result_Met2_Pattern_Ses13_train" folder containes the result of Verfication method two using session one and three for training and session two for testing
	7.1 Data_Pattern_Verfi_Meth2.csv file contains the values of each testers of: Accuracy,	TP,	TN,	FP,	FN,	ERR,	FRR,	FAR,	EER,	Sensitivity(TPR),	TAR,	Specificity(TNR),	
	F-score,	k values,	and the number of Iterations for each tester

 	7.1 tempAver_PatternM2.csv file contains the average result of: global Accuracy,	global Sensitivity,	global Specificity,	global FRR,	global FAR,	global TAR,	global ERR, 	
	global F_score and the	k values with total time taken for each K value features selected for overall testers.
 

 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
8. The  "Result_Met2_Pattern_Ses23_train" folder containes the result of Verfication method two using session two and three for training and session one for testing
	8.1 Data_Pattern_Verfi_Meth2.csv file contains the values of each testers of: Accuracy,	TP,	TN,	FP,	FN,	ERR,	FRR,	FAR,	EER,	Sensitivity(TPR),	TAR,	Specificity(TNR),	
	F-score,	k values,	and the number of Iterations for each tester

 	8.1 tempAver_PatternM2.csv file contains the average result of: global Accuracy,	global Sensitivity,	global Specificity,	global FRR,	global FAR,	global TAR,	global ERR, 	
	global F_score and the	k values with total time taken for each K value features selected for overall testers.
 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

9. The "Roc_Graph_Patterns_test_size_0.2_EER" folder contain the graphs of:
	9.1 Graphs of all testers using four classifiers
	9.2 Micro and Macro averaging graphs
	9.3 The DET curve for Tester 1, 20 and 30.
	9.4 the ROC AUC for tester 1, 20 and 30 using the four classifiers

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

10. Experiment1_Pattern.py file is the programming for preparing dataset one of the session one experiment.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

11. Experiment2_Pattern.py file is the programming for preparing dataset two of the session two experiment.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

12. Experiment3_Pattern.py file is the programming for preparing dataset three of the session three experiment.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

13. Pattern_GridSearch_BeforeTransfromation.py file is the programming code Using SVM, Random Forest, Decision Tree and NN classifer with out applying any transformations.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

14. Pattern_GridSearch_PL_MinMax_MaxAbs_Norm_Transfomation.py  file is the programming code Using SVM, Random Forest, Decision Tree and NN classifer on the MinMax Scaler, Max Absolute Scaler, Robust and Normalizer Data preprocessing.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

15. Pattern_GridSearch_PL_MinMax_transormation_KBest_BestClassifier.py file is the programming code Using SVM and Random Forest Classifier with MinMax Preprocessing using different features selected with KBest feature selector.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

16. PatternsGSearch_PL_Verfication1_Ses12_train.py file is the programming code Using Verfication Method one using SVM with MinMax Preprocessing using session one and two for training and session three for testing with different features selected with KBest feature selector.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

17. PatternsGSearch_PL_Verfication2_Ses12_train.py file is the programming code Using Verfication Method two using SVM with MinMax Preprocessing using session one and two for training and session three for testing with different features selected with KBest feature selector.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

18. PatternsGSearch_PL_Verfication2_Ses13_train.py file is the programming code Using Verfication Method two using SVM with MinMax Preprocessing using session one and three for training and session two for testing with different features selected with KBest feature selector.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

19. PatternsGSearch_PL_Verfication2_Ses23_train.py file is the programming code Using Verfication Method two using SVM with MinMax Preprocessing using session two and three for training and session three for testing with different features selected with KBest feature selector.

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<....................======================================......................>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

20. Patterns_RocGraphAnalysis.py file is the programming code Using the classifer SVM, RF, NN, and DT to draw ROC Curve, Micro and Macro averaging, DET Curves and to compute EER.
