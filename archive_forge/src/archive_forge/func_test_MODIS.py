import numpy as np
from numpy.testing import assert_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_MODIS(self):
    crs = ccrs.Sinusoidal.MODIS
    lons = np.array([-180, -50, 40, 180])
    lats = np.array([-89.999, 30, 20, 89.999])
    expected_x = np.array([-349.33, -4814886.99, 4179566.79, 349.33])
    expected_y = np.array([-10007443.48, 3335851.56, 2223901.04, 10007443.48])
    assert_almost_equal(crs.transform_points(crs.as_geodetic(), lons, lats), np.c_[expected_x, expected_y, [0, 0, 0, 0]], decimal=2)