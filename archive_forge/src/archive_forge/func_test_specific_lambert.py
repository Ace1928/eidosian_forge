from numpy.testing import assert_array_almost_equal
import pyproj
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_specific_lambert():
    crs = ccrs.LambertConformal(central_longitude=10, standard_parallels=(35, 65), central_latitude=52, false_easting=4000000, false_northing=2800000, globe=ccrs.Globe(ellipse='GRS80'))
    other_args = {'ellps=GRS80', 'lon_0=10', 'lat_0=52', 'x_0=4000000', 'y_0=2800000', 'lat_1=35', 'lat_2=65'}
    check_proj_params('lcc', crs, other_args)