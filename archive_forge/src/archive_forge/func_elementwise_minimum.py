from scipy import sparse
import numbers
import numpy as np
def elementwise_minimum(X, Y):
    return if_sparse(sparse_minimum, np.minimum, X, Y)