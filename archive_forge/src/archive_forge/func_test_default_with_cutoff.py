from numpy.testing import assert_array_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_default_with_cutoff():
    crs = ccrs.LambertConformal(cutoff=-80)
    crs2 = ccrs.LambertConformal(cutoff=-80)
    default = ccrs.LambertConformal()
    other_args = {'ellps=WGS84', 'lon_0=-96.0', 'lat_0=39.0', 'x_0=0.0', 'y_0=0.0', 'lat_1=33', 'lat_2=45'}
    check_proj_params('lcc', crs, other_args)
    assert crs == crs2
    assert crs != default
    assert hash(crs) != hash(default)
    assert hash(crs) == hash(crs2)
    assert_array_almost_equal(crs.y_limits, (-49788019.81831982, 30793476.08487709))