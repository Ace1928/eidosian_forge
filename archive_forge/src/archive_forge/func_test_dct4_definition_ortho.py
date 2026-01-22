from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dct4_definition_ortho(mdata_x, rdt):
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    y = dct(x, norm='ortho', type=4)
    y2 = naive_dct4(x, norm='ortho')
    dec = dec_map[dct, rdt, 4]
    assert_equal(y.dtype, dt)
    assert_allclose(y, y2, rtol=0.0, atol=np.max(y2) * 10 ** (-dec))