def clean(X, y):
    # get external support
    import pandas as pd

    """
    TODO:
    Part 0, Step 2: 
        - Use the pandas {isna} and {dropna} functions to remove from the dataset any corrupted samples
    """
    nan_indices_X = X.isna().any(axis=1)

    # Dropping NaN values from X and y
    X = X.dropna()
    y = y[~nan_indices_X]

    # return the cleaned data
    return [X, y]


def train_test_validation_split(X, y, test_size, cv_size):
    # get external support
    from sklearn.model_selection import train_test_split

    """
    TODO:
    Part 0, Step 3: 
        - Use the sklearn {train_test_split} function to split the dataset (and the labels) into
            train, test and cross-validation sets
    """
    test_cv_size = test_size+cv_size

    # split data into train and test - cross validation subsets
    X_train, X_testcv, y_train, y_testcv = train_test_split(
        X, y, test_size=test_cv_size, random_state=0, shuffle=True)

    # split test - cross validation sets into test and cross validation subsets
    X_test, X_cv, y_test, y_cv = train_test_split(
        X_testcv, y_testcv, test_size=cv_size/test_cv_size, random_state=0, shuffle=True)

    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]


def scale(X_train, X_test, X_cv):
    # get external support
    from sklearn import preprocessing

    """
    TODO:
    Part 0, Step 4: 
        - Use the {preprocessing.StandardScaler} of sklearn to normalize the data
        - Scale the train, test and cross-validation sets accordingly
    """
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_cv = scaler.transform(X_cv)

    # return the normalized data and the scaler
    return [X_train, X_test, X_cv, scaler]


def clean_split_scale(X, y):
    # clean data (remove NaN data points)
    [X, y] = clean(X, y)

    # split data into 80% train, 10% test, 10% cross validation
    [X_train, y_train, X_test, y_test, X_cv, y_cv] = train_test_validation_split(
        X, y, test_size=0.1, cv_size=0.1)

    # convert data and labels to numpy arrays, ravel labels
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    X_cv = X_cv.to_numpy()
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    y_cv = y_cv.to_numpy().ravel()

    # scale the data
    [X_train, X_test, X_cv, scaler] = scale(X_train, X_test, X_cv)

    # return cleaned, scaled and split data and the scaler
    return [X_train, y_train, X_test, y_test, X_cv,y_cv,scaler]