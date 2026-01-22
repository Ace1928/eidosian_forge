from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('func', [dct, dctn, idct, idctn, dst, dstn, idst, idstn])
def test_swapped_byte_order(func):
    rng = np.random.RandomState(1234)
    x = rng.rand(10)
    swapped_dt = x.dtype.newbyteorder('S')
    assert_allclose(func(x.astype(swapped_dt)), func(x))