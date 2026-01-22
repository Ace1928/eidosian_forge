from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_easting_northing():
    false_easting = 1000000
    false_northing = -2000000
    crs = ccrs.Mercator(false_easting=false_easting, false_northing=false_northing)
    other_args = {'ellps=WGS84', 'lon_0=0.0', f'x_0={false_easting}', f'y_0={false_northing}', 'units=m'}
    check_proj_params('merc', crs, other_args)
    assert_almost_equal(crs.boundary.bounds, [-19037508, -17496571, 21037508, 16764656], decimal=0)