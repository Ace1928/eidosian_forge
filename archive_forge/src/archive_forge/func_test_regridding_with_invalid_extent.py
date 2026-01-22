import numpy as np
from numpy.testing import assert_array_equal
import pytest
from cartopy.tests.conftest import (
import cartopy.crs as ccrs
import cartopy.img_transform as img_trans
@pytest.mark.parametrize('target_prj', (ccrs.Mollweide(), ccrs.Orthographic()))
@pytest.mark.parametrize('use_scipy', (pytest.param(True, marks=requires_scipy), pytest.param(False, marks=requires_pykdtree)))
def test_regridding_with_invalid_extent(target_prj, use_scipy, monkeypatch):
    lats = np.array([65, 10, -45])
    lons = np.array([-170, 10, 170])
    data = np.array([1, 2, 3])
    data_trans = ccrs.Geodetic()
    target_x, target_y, extent = img_trans.mesh_projection(target_prj, 8, 4)
    if use_scipy:
        monkeypatch.setattr(img_trans, '_is_pykdtree', False)
        import scipy.spatial
        monkeypatch.setattr(img_trans, '_kdtreeClass', scipy.spatial.cKDTree)
    _ = img_trans.regrid(data, lons, lats, data_trans, target_prj, target_x, target_y)