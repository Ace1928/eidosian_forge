import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_syrk(self):
    for f in _get_func('syrk'):
        c = f(a=self.a, alpha=1.0)
        assert_array_almost_equal(np.triu(c), np.triu(self.t))
        c = f(a=self.a, alpha=1.0, lower=1)
        assert_array_almost_equal(np.tril(c), np.tril(self.t))
        c0 = np.ones(self.t.shape)
        c = f(a=self.a, alpha=1.0, beta=1.0, c=c0)
        assert_array_almost_equal(np.triu(c), np.triu(self.t + c0))
        c = f(a=self.a, alpha=1.0, trans=1)
        assert_array_almost_equal(np.triu(c), np.triu(self.tt))