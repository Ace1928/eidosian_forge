from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('rdt', [np.longdouble, np.float64, np.float32, int])
def test_dst1_definition_ortho(rdt, mdata_x):
    dec = dec_map[dst, rdt, 1]
    x = np.array(mdata_x, dtype=rdt)
    dt = np.result_type(np.float32, rdt)
    y = dst(x, norm='ortho', type=1)
    y2 = naive_dst1(x, norm='ortho')
    assert_equal(y.dtype, dt)
    assert_allclose(y, y2, rtol=0.0, atol=np.max(y2) * 10 ** (-dec))