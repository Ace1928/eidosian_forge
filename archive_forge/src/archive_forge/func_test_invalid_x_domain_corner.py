import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import cartopy.crs as ccrs
def test_invalid_x_domain_corner(self):
    rlon = np.array([180.0])
    rlat = np.array([90.0])
    u = np.array([1.0])
    v = np.array([-1.0])
    src_proj = ccrs.PlateCarree()
    target_proj = ccrs.Stereographic(central_latitude=90, central_longitude=0)
    with pytest.warns(UserWarning):
        warnings.simplefilter('always')
        ut, vt = target_proj.transform_vectors(src_proj, rlon, rlat, u, v)