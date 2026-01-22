import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_sphere_transform():
    globe = ccrs.Globe(semimajor_axis=1.0, semiminor_axis=1.0, ellipse=None)
    ortho = ccrs.Orthographic(central_latitude=40.0, central_longitude=-100.0, globe=globe)
    geodetic = ortho.as_geodetic()
    other_args = {'a=1.0', 'b=1.0', 'lon_0=-100.0', 'lat_0=40.0'}
    check_proj_params('ortho', ortho, other_args)
    assert_almost_equal(np.array(ortho.x_limits), [-0.99999, 0.99999])
    assert_almost_equal(np.array(ortho.y_limits), [-0.99999, 0.99999])
    result = ortho.transform_point(-110.0, 30.0, geodetic)
    assert_almost_equal(result, np.array([-0.1503837, -0.1651911]))
    inverse_result = geodetic.transform_point(result[0], result[1], ortho)
    assert_almost_equal(inverse_result, [-110.0, 30.0])