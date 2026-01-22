import numpy as np
from numpy.testing import assert_almost_equal
import pytest
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_ellipse_globe():
    globe = ccrs.Globe(ellipse='WGS84')
    with pytest.warns(UserWarning, match='does not handle elliptical globes.') as w:
        ortho = ccrs.Orthographic(globe=globe)
        assert len(w) == 1
    other_args = {'ellps=WGS84', 'lon_0=0.0', 'lat_0=0.0'}
    check_proj_params('ortho', ortho, other_args)
    assert_almost_equal(ortho.x_limits, [-6378073.21863, 6378073.21863])
    assert_almost_equal(ortho.y_limits, [-6378073.21863, 6378073.21863])