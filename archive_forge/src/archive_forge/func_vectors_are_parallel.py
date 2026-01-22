import numpy as np
from scipy.sparse.linalg import aslinearoperator
def vectors_are_parallel(v, w):
    if v.ndim != 1 or v.shape != w.shape:
        raise ValueError('expected conformant vectors with entries in {-1,1}')
    n = v.shape[0]
    return np.dot(v, w) == n