# Solution
def PCA_scratch(X, num_components=None):
    """
    Perform PCA on the input data using a manual implementation.

    Parameters:
    X (ndarray): The dataset to perform PCA on.
    num_components (int, optional): The number of principal components to keep. If not specified, all components will be kept.

    Returns:
    X_reduced (ndarray): The dataset projected onto the selected principal components.
    coverage (float): The proportion of variance explained by the selected principal components.
    eigenvalues (ndarray): The eigenvalues of the covariance matrix, sorted in descending order.
    eigenvectors (ndarray): The eigenvectors of the covariance matrix, sorted in descending order by the corresponding eigenvalues.
    """
    # Step 1: Center and scale the data

    X_centered = X - np.mean(X, axis=0)
    X_scaled = X_centered / np.std(X_centered, axis=0)

    # Step 2: Compute the covariance matrix

    covariance_matrix = np.cov(X_scaled, rowvar=False)

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select the top num_components eigenvectors, if None, keep all components
    sum_eigenvalues = np.sum(eigenvalues) # This is for step 7

    if num_components is not None:
        eigenvectors = eigenvectors[:, :num_components]
        eigenvalues = eigenvalues[:num_components]
    
    # Step 6: Transform the data onto the selected principal components

    X_reduced = np.dot(eigenvectors.transpose(), X_centered.transpose()).transpose()

    # Step 7: Compute the proportion of variance explained by the selected principal components

    coverage = np.sum(eigenvalues) / sum_eigenvalues

    return X_reduced, coverage, eigenvalues, eigenvectors
