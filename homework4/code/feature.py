import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    N, d = X.shape
    num_new_features = 3
    
    X_new = np.zeros((N, d + num_new_features))
    X_new[:, :d] = X
    
    # Add new features
    X_new[:, d] = X[:, 0] ** 2
    X_new[:, d+1] = X[:, 1] ** 2
    X_new[:, d+2] = np.sin(X[:, 0] * X[:, 1])

    return X_new
    
    raise NotImplementedError
