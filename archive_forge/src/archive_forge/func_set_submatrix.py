from scipy import sparse
import numbers
import numpy as np
def set_submatrix(X, i, j, values):
    X[np.ix_(i, j)] = values
    return X