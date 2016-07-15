####
#SVM-linear training and prediction
#Calc time and accuracy
####
import sys
from time import time
from sklearn.svm import SVC
sys.path.append("tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)//100] 
labels_train = labels_train[:len(labels_train)//100] 
#########################################################
clf=SVC(kernel="linear")
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
t1 = time()
print("Prediction:", clf.predict(features_test))
print ("prediction time:", round(time()-t1, 3), "s")
print("Accuracy: ",clf.score(features_test, labels_test))
#########################################################