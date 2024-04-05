# @STUDENT: do not change these {import}s
from library import read_dataset, clean_split_scale
from library import apply_pca
from library import evaluate, f1score
from solution import apply_svc

# main function
if __name__ == '__main__':

    # load whole dataset
    [X, y] = read_dataset(labels=[5, 6, 8])

    # cleanup data, split data in training set and test set, normalize data
    [X_train, y_train, X_test, y_test, X_cv, y_cv, scaler] = clean_split_scale(X, y)

    # apply the PCA
    pca = apply_pca(X_train)

    # apply the support vector classifier using the features provided by the PCA 
    clf = apply_svc(pca, X_train, y_train)

    # TODO: apply the principal components transformation to the test and validation sets
    X_test = pca.transform(X_test)
    X_cv = pca.transform(X_cv)
    # TODO: make predictions on the validation set
    y_predicted_cv = clf.predict(X_cv)

    # TODO: make predictions on the test set
    y_predicted_test = clf.predict(X_test)

    # TODO: assess performance on the validation and test set

    # HINT: performance reports can be obtained by appropriately editing the following lines:
    print(f"\n\nF1-score on the validation set is {f1score(y_cv, y_predicted_cv)}\n")
    evaluate(clf, y_cv, y_predicted_cv)

    print(f"\n\nF1-score on the validation set is {f1score(y_test, y_predicted_test)}\n")
    evaluate(clf, y_test, y_predicted_test)
