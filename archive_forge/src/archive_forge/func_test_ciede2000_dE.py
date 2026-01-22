import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color.delta_e import (
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_ciede2000_dE(dtype, channel_axis):
    data = load_ciede2000_data()
    N = len(data)
    lab1 = np.zeros((N, 3), dtype=dtype)
    lab1[:, 0] = data['L1']
    lab1[:, 1] = data['a1']
    lab1[:, 2] = data['b1']
    lab2 = np.zeros((N, 3), dtype=dtype)
    lab2[:, 0] = data['L2']
    lab2[:, 1] = data['a2']
    lab2[:, 2] = data['b2']
    lab1 = np.moveaxis(lab1, source=-1, destination=channel_axis)
    lab2 = np.moveaxis(lab2, source=-1, destination=channel_axis)
    dE2 = deltaE_ciede2000(lab1, lab2, channel_axis=channel_axis)
    assert dE2.dtype == _supported_float_type(dtype)
    assert_allclose(dE2, data['dE'], rtol=0.01)