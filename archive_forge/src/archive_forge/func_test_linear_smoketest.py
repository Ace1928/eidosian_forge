import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_linear_smoketest(self):
    funcs = [lambda x, y: 0 * x + 1, lambda x, y: 0 + x, lambda x, y: -2 + y, lambda x, y: 3 + 3 * x + 14.15 * y]
    for j, func in enumerate(funcs):
        self._check_accuracy(func, tol=1e-13, atol=1e-07, rtol=1e-07, err_msg='Function %d' % j)
        self._check_accuracy(func, tol=1e-13, atol=1e-07, rtol=1e-07, alternate=True, err_msg='Function (alternate) %d' % j)
        self._check_accuracy(func, tol=1e-13, atol=1e-07, rtol=1e-07, err_msg='Function (rescaled) %d' % j, rescale=True)
        self._check_accuracy(func, tol=1e-13, atol=1e-07, rtol=1e-07, alternate=True, rescale=True, err_msg='Function (alternate, rescaled) %d' % j)