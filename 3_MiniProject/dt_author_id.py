 
'''

change the percentile in email_preprocessor and then you would realize the diff in accuracy and the time to train./
''' 
import sys
from time import time
sys.path.append("tools/")
from sklearn import tree
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#########################################################
### your code goes here ###

clf=tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
print ("Training Time: ", round(time()-t0, 3), "s")

pred=clf.predict(features_test, labels_test)
accuracy = clf.score( features_test, labels_test)
print(accuracy)

#########################################################

print(len(features_train[0]))

