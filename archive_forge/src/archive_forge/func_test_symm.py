import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_symm(self):
    for f in _get_func('symm'):
        res = f(a=self.a, b=self.b, c=self.c, alpha=1.0, beta=1.0)
        assert_array_almost_equal(res, self.t)
        res = f(a=self.a.T, b=self.b, lower=1, c=self.c, alpha=1.0, beta=1.0)
        assert_array_almost_equal(res, self.t)
        res = f(a=self.a, b=self.b.T, side=1, c=self.c.T, alpha=1.0, beta=1.0)
        assert_array_almost_equal(res, self.t.T)