import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_descending_points(self):

    def value_func_4d(x, y, z, a):
        return 2 * x ** 3 + 3 * y ** 2 - z - a
    x1 = np.array([0, 1, 2, 3])
    x2 = np.array([0, 10, 20, 30])
    x3 = np.array([0, 10, 20, 30])
    x4 = np.array([0, 0.1, 0.2, 0.3])
    points = (x1, x2, x3, x4)
    values = value_func_4d(*np.meshgrid(*points, indexing='ij', sparse=True))
    pts = (0.1, 0.3, np.transpose(np.linspace(0, 30, 4)), np.linspace(0, 0.3, 4))
    correct_result = interpn(points, values, pts)
    x1_descend = x1[::-1]
    x2_descend = x2[::-1]
    x3_descend = x3[::-1]
    x4_descend = x4[::-1]
    points_shuffled = (x1_descend, x2_descend, x3_descend, x4_descend)
    values_shuffled = value_func_4d(*np.meshgrid(*points_shuffled, indexing='ij', sparse=True))
    test_result = interpn(points_shuffled, values_shuffled, pts)
    assert_array_equal(correct_result, test_result)