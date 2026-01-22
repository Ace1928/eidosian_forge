import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def check_ranks(a):
    for method in ('min', 'max', 'dense', 'ordinal', 'average'):
        out = rankdata(a, method=method)
        assert_array_equal(out, rankf[method](a))