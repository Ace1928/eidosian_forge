import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
@pytest.mark.parametrize('lat', [-10, 0, 10])
@pytest.mark.parametrize('lon', [-10, 0, 10])
def test_central_params(lat, lon):
    ortho = ccrs.Orthographic(central_latitude=lat, central_longitude=lon)
    other_args = {f'lat_0={lat}', f'lon_0={lon}', 'a=6378137.0'}
    check_proj_params('ortho', ortho, other_args)
    assert_almost_equal(np.array(ortho.x_limits), [-6378073.21863, 6378073.21863])
    assert_almost_equal(np.array(ortho.y_limits), [-6378073.21863, 6378073.21863])