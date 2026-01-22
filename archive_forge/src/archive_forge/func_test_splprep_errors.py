import os
import operator
import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
import scipy.linalg as sl
from scipy.interpolate._bsplines import (_not_a_knot, _augknt,
import scipy.interpolate._fitpack_impl as _impl
from scipy._lib._util import AxisError
def test_splprep_errors(self):
    x = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    with assert_raises(ValueError, match='too many values to unpack'):
        splprep(x)
    with assert_raises(ValueError, match='too many values to unpack'):
        _impl.splprep(x)
    x = np.linspace(0, 40, num=3)
    with assert_raises(TypeError, match='m > k must hold'):
        splprep([x])
    with assert_raises(TypeError, match='m > k must hold'):
        _impl.splprep([x])
    x = [-50.49072266, -50.49072266, -54.49072266, -54.49072266]
    with assert_raises(ValueError, match='Invalid inputs'):
        splprep([x])
    with assert_raises(ValueError, match='Invalid inputs'):
        _impl.splprep([x])
    x = [1, 3, 2, 4]
    u = [0, 0.3, 0.2, 1]
    with assert_raises(ValueError, match='Invalid inputs'):
        splprep(*[[x], None, u])