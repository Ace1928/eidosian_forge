import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_ellipsoid_transform():
    globe = ccrs.Globe(ellipse='clrk66')
    utm = ccrs.UTM(zone=18, globe=globe)
    geodetic = utm.as_geodetic()
    other_args = {'ellps=clrk66', 'units=m', 'zone=18'}
    check_proj_params('utm', utm, other_args)
    assert_almost_equal(np.array(utm.x_limits), [-250000, 1250000])
    assert_almost_equal(np.array(utm.y_limits), [-10000000, 25000000])
    result = utm.transform_point(-73.5, 40.5, geodetic)
    assert_almost_equal(result, np.array([127106.5 + 500000, 4484124.4]), decimal=1)
    inverse_result = geodetic.transform_point(result[0], result[1], utm)
    assert_almost_equal(inverse_result, [-73.5, 40.5])