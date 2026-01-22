import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
from skimage.color.delta_e import (
@pytest.mark.parametrize('channel_axis', [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cmc(dtype, channel_axis):
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
    dE2 = deltaE_cmc(lab1, lab2, channel_axis=channel_axis)
    assert dE2.dtype == _supported_float_type(dtype)
    oracle = np.array([1.73873611, 2.49660844, 3.30494501, 0.85735576, 0.88332927, 0.97822692, 3.50480874, 2.87930032, 6.5783807, 6.57838075, 6.5783808, 6.57838086, 6.67492321, 6.67492326, 6.67492331, 4.66852997, 42.10875485, 39.45889064, 38.36005919, 33.93663807, 1.14400168, 1.00600419, 1.11302547, 1.05335328, 1.42822951, 1.2548143, 1.76838061, 2.02583367, 3.08695508, 1.74893533, 1.90095165, 1.70258148, 1.80317207, 2.44934417])
    rtol = 1e-05 if dtype == np.float32 else 1e-08
    assert_allclose(dE2, oracle, rtol=rtol)
    lab1 = lab2
    expected = np.zeros_like(oracle)
    assert_almost_equal(deltaE_cmc(lab1, lab2, channel_axis=channel_axis), expected, decimal=6)
    lab2[0, 0] += np.finfo(float).eps
    assert_almost_equal(deltaE_cmc(lab1, lab2, channel_axis=channel_axis), expected, decimal=6)