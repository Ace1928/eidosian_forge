import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_syrk_wrong_c(self):
    f = getattr(fblas, 'dsyrk', None)
    if f is not None:
        assert_raises(Exception, f, **{'a': self.a, 'alpha': 1.0, 'c': np.ones((5, 8))})