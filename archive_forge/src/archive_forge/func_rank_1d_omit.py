import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def rank_1d_omit(a, method):
    out = np.zeros_like(a)
    i = np.isnan(a)
    a_compressed = a[~i]
    res = rankdata(a_compressed, method)
    out[~i] = res
    out[i] = np.nan
    return out