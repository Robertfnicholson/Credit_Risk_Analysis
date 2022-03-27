# Credit Risk Analysis Challenge – 
## Overview of Project
This challenge principally involved the use of supervised machine learning models from Imbalanced-learn 
and Scikit-learn libraries and Python to predict credit risk. I employed different techniques to train 
and evaluate models with unbalanced classes. Using the credit card credit dataset from LendingClub, a 
peer-to-peer lending services company, I oversampled the data using the RandomOverSampler and SMOTE 
algorithms, and then undersampled the data using the ClusterCentroids algorithm. Then, I used a 
combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I compared two new 
machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to 
predict credit risk. Finally, I evaluated the performance of these models. My code can be found in 
“credit_risk_resampling.ipynb” and “credit_risk_ensemble.ipynb.” </p>

## Results
The below table summarizes my results, showing the balanced accuracy score and the precision and recall 
scores of all six machine learning models in predicting credit risk. The balanced accuracy score is a 
measure of each model’s performance based on the difference between its predicted values and actual values. 
The precision score is a measure of how reliable a positive classification is. It is calculated by dividing 
the number of true predicted values (TP) by the number of all positives, i.e. the sum of true positives and 
false positives (FP). In other words, the precision score is equal to TP/(TP +FP). The sensitivity or recall 
score is a measure of our model correctly identifying true positives (TP). It is calculated by dividing the 
number of true predicted positives (TP) by the number of all true positives (TP) and false negatives (FN). 
In other words, the sensitivity score is equal to TP/(TP +FN). </p>

![Machine_learning_model_table.png]( https://github.com/Robertfnicholson/Credit_Risk_Analysis/blob/d4ccaa868e33944b5b774ea9bf0b26fda73a554b/Machine_learning_model_table.png) 

*	The focus of each model is predicting if a loan will go bad, i.e. in default. As such, the sensitivity or 
	recall score is critical, and the balance accuracy score is also helpful.
*	The challenge focused on the smaller class of high-risk accounts and predicting which of these accounts 
	would default.
*	As the table above indicates, both the Balanced Random Forest Classifier and the Easy Ensemble AdaBoost 
	Classifier models achieved the highest recall score of 0.72 for the high-risk accounts and the highest balanced 
	accuracy score of 0.806. In addition, these models had the highest recall score of 0.89 for the low-risk accounts. 
*	Both the Naive Random Oversampling and Undersampling using Cluster Centroids models generated the lowest 
	recall scores for high-risk accounts of 0.52. 
*	The Combination Sampling model generated the lowest balanced accuracy score. 
*	The SMOTE Oversampling model generated the lowest recall score for the low-risk accounts. 
</p>

## Summary
Six machine learning models were used to predict bad loans for Lending Club. The Balanced Random Forest Classifier 
and the Easy Ensemble AdaBoost Classifier models achieved the highest scores for predicting bad loans as evidenced 
by their recall scores and balanced accuracy scores. Since the Easy Ensemble AdaBoost Classifier model did not 
generate an incremental increase in predictive value over the Balanced Random Forest Classifier model, I recommend 
using the Balanced Random Forest Classifier model. 

