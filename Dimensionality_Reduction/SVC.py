# @STUDENT: do not change these {import}s
# support for SVM
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

def apply_svc(pca, X_train, y_train):

    # TODO: apply the principal components transformation to the training set
    # Z_train = ...
    Z_train = pca.transform(X_train)
    gamma  = 0.001
    C =100
    # TODO: create a one-vs-rest support vector classifier
    clf = OneVsRestClassifier(svm.SVC(C=C,kernel='rbf', gamma=gamma))
    clf.fit(Z_train, y_train)


    # TODO: train the classifier on the newly constructed features

    # all done
    return clf
