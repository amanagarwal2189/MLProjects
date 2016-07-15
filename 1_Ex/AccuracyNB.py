def accuracy(clf,  features_test, labels_test):
    accuracy = clf.score(features_test, labels_test, None)
    return accuracy