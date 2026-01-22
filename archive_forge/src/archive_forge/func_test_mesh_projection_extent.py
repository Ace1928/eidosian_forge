import numpy as np
from numpy.testing import assert_array_equal
import pytest
from cartopy.tests.conftest import (
import cartopy.crs as ccrs
import cartopy.img_transform as img_trans
@pytest.mark.parametrize('xmin, xmax', [(-90, 0), (-90, 90), (-90, None), (0, 90), (0, None), (None, 0), (None, 90), (None, None)])
@pytest.mark.parametrize('ymin, ymax', [(-45, 0), (-45, 45), (-45, None), (0, 45), (0, None), (None, 0), (None, 45), (None, None)])
def test_mesh_projection_extent(xmin, xmax, ymin, ymax):
    proj = ccrs.PlateCarree()
    nx = 4
    ny = 2
    target_x, target_y, extent = img_trans.mesh_projection(proj, nx, ny, x_extents=(xmin, xmax), y_extents=(ymin, ymax))
    if xmin is None:
        xmin = proj.x_limits[0]
    if xmax is None:
        xmax = proj.x_limits[1]
    if ymin is None:
        ymin = proj.y_limits[0]
    if ymax is None:
        ymax = proj.y_limits[1]
    assert_array_equal(extent, [xmin, xmax, ymin, ymax])
    assert_array_equal(np.diff(target_x, axis=1), (xmax - xmin) / nx)
    assert_array_equal(np.diff(target_y, axis=0), (ymax - ymin) / ny)