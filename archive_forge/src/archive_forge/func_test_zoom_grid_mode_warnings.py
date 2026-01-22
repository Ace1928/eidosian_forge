import sys
import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import scipy.ndimage as ndimage
from . import types
@pytest.mark.parametrize('mode', ['constant', 'wrap'])
def test_zoom_grid_mode_warnings(self, mode):
    x = numpy.arange(9, dtype=float).reshape((3, 3))
    with pytest.warns(UserWarning, match='It is recommended to use mode'):
        (ndimage.zoom(x, 2, mode=mode, grid_mode=True),)