import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def average_rank(a):
    return [(i + j) / 2.0 for i, j in zip(min_rank(a), max_rank(a))]