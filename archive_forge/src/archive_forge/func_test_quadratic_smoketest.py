import os
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises
import pytest
import scipy.interpolate.interpnd as interpnd
import scipy.spatial._qhull as qhull
import pickle
def test_quadratic_smoketest(self):
    funcs = [lambda x, y: x ** 2, lambda x, y: y ** 2, lambda x, y: x ** 2 - y ** 2, lambda x, y: x * y]
    for j, func in enumerate(funcs):
        self._check_accuracy(func, tol=1e-09, atol=0.22, rtol=0, err_msg='Function %d' % j)
        self._check_accuracy(func, tol=1e-09, atol=0.22, rtol=0, err_msg='Function %d' % j, rescale=True)