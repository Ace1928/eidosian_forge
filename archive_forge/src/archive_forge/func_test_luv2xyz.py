import colorsys
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage import data
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage._shared.utils import _supported_float_type, slice_at_axis
from skimage.color import (
from skimage.util import img_as_float, img_as_ubyte, img_as_float32
def test_luv2xyz(self):
    assert_array_almost_equal(luv2xyz(self.luv_array), self.xyz_array, decimal=3)
    for I in ['A', 'B', 'C', 'd50', 'd55', 'd65']:
        I = I.lower()
        for obs in ['2', '10', 'R']:
            obs = obs.lower()
            fname = f'color/tests/data/luv_array_{I}_{obs}.npy'
            luv_array_I_obs = np.load(fetch(fname))
            assert_array_almost_equal(luv2xyz(luv_array_I_obs, I, obs), self.xyz_array, decimal=3)
    for I in ['d75', 'e']:
        fname = f'color/tests/data/luv_array_{I}_2.npy'
        luv_array_I_obs = np.load(fetch(fname))
        assert_array_almost_equal(luv2xyz(luv_array_I_obs, I, '2'), self.xyz_array, decimal=3)