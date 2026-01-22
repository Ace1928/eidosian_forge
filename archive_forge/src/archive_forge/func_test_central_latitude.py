from numpy.testing import assert_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_central_latitude():
    geos = ccrs.NearsidePerspective(central_latitude=53.7)
    other_args = {'a=6378137.0', 'h=35785831', 'lat_0=53.7', 'lon_0=0.0', 'units=m', 'x_0=0', 'y_0=0'}
    check_proj_params('nsper', geos, other_args)
    assert_almost_equal(geos.boundary.bounds, (-5476336.098, -5476336.098, 5476336.098, 5476336.098), decimal=3)