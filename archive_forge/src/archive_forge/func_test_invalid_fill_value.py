import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_invalid_fill_value(self):
    np.random.seed(1234)
    x = np.linspace(0, 2, 5)
    y = np.linspace(0, 1, 7)
    values = np.random.rand(5, 7)
    RegularGridInterpolator((x, y), values, fill_value=1)
    assert_raises(ValueError, RegularGridInterpolator, (x, y), values, fill_value=1 + 2j)