import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
def test_args_gh14589(self):
    """Using `args` used to cause `shgo` to fail; see #14589, #15986,
        #16506"""
    res = shgo(func=lambda x, y, z: x * z + y, bounds=[(0, 3)], args=(1, 2))
    ref = shgo(func=lambda x: 2 * x + 1, bounds=[(0, 3)])
    assert_allclose(res.fun, ref.fun)
    assert_allclose(res.x, ref.x)