from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
@pytest.mark.parametrize('lon', [-10.0, 10.0])
def test_central_longitude(lon):
    crs = ccrs.Mercator(central_longitude=lon)
    other_args = {'ellps=WGS84', f'lon_0={lon}', 'x_0=0.0', 'y_0=0.0', 'units=m'}
    check_proj_params('merc', crs, other_args)
    assert_almost_equal(crs.boundary.bounds, [-20037508, -15496570, 20037508, 18764656], decimal=0)