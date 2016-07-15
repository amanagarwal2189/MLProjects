####
#SVM-rbf training and prediction with small dataset
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
clf=SVC(kernel="rbf", C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
t1 = time()
pred=clf.predict(features_test)
print("Prediction:", pred)
print ("prediction time:", round(time()-t1, 3), "s")
print("Accuracy: ",clf.score(features_test, labels_test))
print("Prediction for 10th, 26th and 50th element", pred[10], pred[26], pred[50] )
#########################################################

"""
with only kernel="rbf"

no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 0.328 s
Prediction: [0 1 1 ..., 1 1 1]
prediction time: 3.016 s
Accuracy:  0.616040955631
"""

"""
with only kernel="rbf", C=10.0

no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 0.328 s
Prediction: [0 1 1 ..., 1 1 1]
prediction time: 3.049 s
Accuracy:  0.616040955631
"""

"""
with only kernel="rbf", C=100.0
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 0.328 s
Prediction: [0 1 1 ..., 1 1 1]
prediction time: 3.047 s
Accuracy:  0.616040955631
"""
"""
with only kernel="rbf", C=1000.0
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 0.302 s
Prediction: [0 1 1 ..., 1 0 1]
prediction time: 2.891 s
Accuracy:  0.821387940842
"""
"""
with only kernel="rbf", C=10000.0

no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 0.281 s
Prediction: [0 1 1 ..., 1 0 1]
prediction time: 2.442 s
Accuracy:  0.892491467577
"""