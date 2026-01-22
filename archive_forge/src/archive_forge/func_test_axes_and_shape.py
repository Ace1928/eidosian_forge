from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
@pytest.mark.parametrize('fforward,finverse', [(dctn, idctn), (dstn, idstn)])
def test_axes_and_shape(self, fforward, finverse):
    with assert_raises(ValueError, match='when given, axes and shape arguments have to be of the same length'):
        fforward(self.data, s=self.data.shape[0], axes=(0, 1))
    with assert_raises(ValueError, match='when given, axes and shape arguments have to be of the same length'):
        fforward(self.data, s=self.data.shape, axes=0)