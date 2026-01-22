import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.interpolate import (griddata, NearestNDInterpolator,
def test_multivalue_2d(self):
    x = np.array([(0, 0), (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.25, 0.3)], dtype=np.float64)
    y = np.arange(x.shape[0], dtype=np.float64)[:, None] + np.array([0, 1])[None, :]
    for method in ('nearest', 'linear', 'cubic'):
        for rescale in (True, False):
            msg = repr((method, rescale))
            yi = griddata(x, y, x, method=method, rescale=rescale)
            assert_allclose(y, yi, atol=1e-14, err_msg=msg)