# @STUDENT: do not change these {import}s
# general purpose support
import numpy as np
# support for PCA
from sklearn.decomposition import PCA

def apply_pca(X_train):

    # TODO: apply the sklearn {PCA} with 150 principal components 
    n_components = 150
    pca = PCA(n_components)
    X_transform = pca.fit_transform(X_train)
    # get the explained variance of each of the {n_components} components
    explained_variance_components = pca.explained_variance_ratio_
    
    # TODO: modify the line below to calculate the total {explained_variance} 
    # of all components
    explained_variance = np.cumsum(explained_variance_components)*100

    # report
    print(f"Explained variance = {explained_variance}")

    # return the fitted {pca}
    return pca
