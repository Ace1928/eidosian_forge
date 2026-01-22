import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_eastings(self):
    aeqd_offset = ccrs.AzimuthalEquidistant(false_easting=1234, false_northing=-4321)
    other_args = {'ellps=WGS84', 'lon_0=0.0', 'lat_0=0.0', 'x_0=1234', 'y_0=-4321'}
    check_proj_params('aeqd', aeqd_offset, other_args)
    assert_almost_equal(np.array(aeqd_offset.x_limits), [-20036274.34278924, 20038742.34278924], decimal=6)
    assert_almost_equal(np.array(aeqd_offset.y_limits), [-19974647.371123, 19966005.371123], decimal=6)