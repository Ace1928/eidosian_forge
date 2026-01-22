import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_shuffle_on_ndim_equals_three():

    def to_tuple(A):
        return tuple((tuple((tuple(C) for C in B)) for B in A))
    A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    S = set(to_tuple(A))
    shuffle(A)
    assert set(to_tuple(A)) == S