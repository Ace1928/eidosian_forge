import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_shapes_1d_vectorvalue(self):
    P = BarycentricInterpolator(self.xs, np.outer(self.ys, [1]))
    assert_array_equal(np.shape(P(0)), (1,))
    assert_array_equal(np.shape(P([0])), (1, 1))
    assert_array_equal(np.shape(P([0, 1])), (2, 1))