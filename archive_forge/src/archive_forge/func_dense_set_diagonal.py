from scipy import sparse
import numbers
import numpy as np
def dense_set_diagonal(X, diag):
    X[np.diag_indices(X.shape[0])] = diag
    return X