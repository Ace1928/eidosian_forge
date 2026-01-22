from scipy import sparse
import numbers
import numpy as np
def matrix_is_equivalent(X, Y):
    """
    Checks matrix equivalence with numpy, scipy and pandas
    """
    return X is Y or (isinstance(X, Y.__class__) and X.shape == Y.shape and (np.sum((X != Y).sum()) == 0))