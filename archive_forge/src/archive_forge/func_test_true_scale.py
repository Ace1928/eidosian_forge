import numpy as np
from numpy.testing import assert_almost_equal
import cartopy.crs as ccrs
from .helpers import check_proj_params
def test_true_scale(self):
    globe = ccrs.Globe(ellipse='sphere')
    stereo = ccrs.NorthPolarStereo(true_scale_latitude=30, globe=globe)
    other_args = {'ellps=sphere', 'lat_0=90', 'lon_0=0.0', 'lat_ts=30', 'x_0=0.0', 'y_0=0.0'}
    check_proj_params('stere', stereo, other_args)