import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@pytest.mark.parametrize('fill_value', [None, np.nan, np.pi])
@pytest.mark.parametrize('method', ['linear', 'nearest'])
def test_length_one_axis2(self, fill_value, method):
    options = {'fill_value': fill_value, 'bounds_error': False, 'method': method}
    x = np.linspace(0, 2 * np.pi, 20)
    z = np.sin(x)
    fa = RegularGridInterpolator((x,), z[:], **options)
    fb = RegularGridInterpolator((x, [0]), z[:, None], **options)
    x1a = np.linspace(-1, 2 * np.pi + 1, 100)
    za = fa(x1a)
    y1b = np.zeros(100)
    zb = fb(np.vstack([x1a, y1b]).T)
    assert_allclose(zb, za)
    y1b = np.ones(100)
    zb = fb(np.vstack([x1a, y1b]).T)
    if fill_value is None:
        assert_allclose(zb, za)
    else:
        assert_allclose(zb, fill_value)