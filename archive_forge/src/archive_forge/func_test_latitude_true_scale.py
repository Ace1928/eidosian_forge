from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_latitude_true_scale():
    lat_ts = 20.0
    crs = ccrs.Mercator(latitude_true_scale=lat_ts)
    other_args = {'ellps=WGS84', 'lon_0=0.0', 'x_0=0.0', 'y_0=0.0', 'units=m', f'lat_ts={lat_ts}'}
    check_proj_params('merc', crs, other_args)
    assert_almost_equal(crs.boundary.bounds, [-18836475, -14567718, 18836475, 17639917], decimal=0)