import numpy as np
from numpy.testing import assert_array_equal
import pytest
from cartopy.tests.conftest import (
import cartopy.crs as ccrs
import cartopy.img_transform as img_trans
def test_gridding_data_std_range():
    target_prj = ccrs.PlateCarree()
    lats = np.array([65, 10, -45])
    lons = np.array([-90, 0, 90])
    data = np.array([1, 2, 3])
    data_trans = ccrs.Geodetic()
    target_x, target_y, extent = img_trans.mesh_projection(target_prj, 8, 4)
    image = img_trans.regrid(data, lons, lats, data_trans, target_prj, target_x, target_y, mask_extrapolated=True)
    expected = np.array([[3, 3, 3, 3, 3, 3, 3, 3], [3, 1, 2, 2, 2, 3, 3, 3], [1, 1, 1, 2, 2, 2, 3, 1], [1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float64)
    expected_mask = np.array([[True, True, True, True, True, True, True, True], [True, False, False, False, False, False, False, True], [True, False, False, False, False, False, False, True], [True, True, True, True, True, True, True, True]])
    assert_array_equal([-180, 180, -90, 90], extent)
    assert_array_equal(expected, image)
    assert_array_equal(expected_mask, image.mask)