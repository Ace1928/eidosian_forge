from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('fforward,finverse', [(dctn, idctn), (dstn, idstn)])
@pytest.mark.parametrize('axes', [None, 1, (1,), [1], 0, (0,), [0], (0, 1), [0, 1], (-2, -1), [-2, -1]])
@pytest.mark.parametrize('dct_type', dct_type)
@pytest.mark.parametrize('norm', ['ortho'])
def test_axes_round_trip(self, fforward, finverse, axes, dct_type, norm):
    tmp = fforward(self.data, type=dct_type, axes=axes, norm=norm)
    tmp = finverse(tmp, type=dct_type, axes=axes, norm=norm)
    assert_array_almost_equal(self.data, tmp, decimal=12)