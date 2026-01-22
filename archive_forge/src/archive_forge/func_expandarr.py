import numpy as np
from scipy import signal
from numpy.testing import assert_array_equal, assert_array_almost_equal
def expandarr(x, k):
    kadd = k
    if np.ndim(x) == 2:
        kadd = (kadd, np.shape(x)[1])
    return np.r_[np.ones(kadd) * x[0], x, np.ones(kadd) * x[-1]]