from numpy.testing import assert_array_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_single_npole(self):
    n_pole_crs = ccrs.LambertConformal(standard_parallels=[1.0])
    expected_x = (-20130569, 20130569)
    expected_y = (-8170229, 726200683)
    if pyproj.__proj_version__ >= '9.2.0':
        expected_x = (-20222156, 20222156)
        expected_y = (-8164817, 360848719)
    assert_array_almost_equal(n_pole_crs.x_limits, expected_x, decimal=0)
    assert_array_almost_equal(n_pole_crs.y_limits, expected_y, decimal=0)