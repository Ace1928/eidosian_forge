import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_ellipsoid_guam_transform(self):
    globe = ccrs.Globe(ellipse=None, semimajor_axis=6378206.4, flattening=1 - np.sqrt(1 - 0.00676866))
    lat_0 = 13 + (28 + 20.87887 / 60) / 60
    lon_0 = 144 + (44 + 55.50254 / 60) / 60
    aeqd = ccrs.AzimuthalEquidistant(central_latitude=lat_0, central_longitude=lon_0, false_easting=50000.0, false_northing=50000.0, globe=globe)
    geodetic = aeqd.as_geodetic()
    other_args = {'a=6378206.4', 'f=0.003390076308689371', 'lon_0=144.7487507055556', 'lat_0=13.47246635277778', 'x_0=50000.0', 'y_0=50000.0'}
    check_proj_params('aeqd', aeqd, other_args)
    assert_almost_equal(np.array(aeqd.x_limits), [-19987726.3693194, 20087726.3693194], decimal=6)
    assert_almost_equal(np.array(aeqd.y_limits), [-19919796.94787477, 20019796.94787477], decimal=6)
    pt_lat = 13 + (20 + 20.53846 / 60) / 60
    pt_lon = 144 + (38 + 7.19265 / 60) / 60
    result = aeqd.transform_point(pt_lon, pt_lat, geodetic)
    assert_array_almost_equal(result, [37712.48, 35242.0], decimal=1)