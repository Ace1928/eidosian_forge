import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.vector_transform as vec_trans
def test_with_scalar_field(self):
    expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
    expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
    expected_u_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
    expected_v_grid = np.array([[np.nan, 0.8, 0.3, 0.8, np.nan], [np.nan, 2.675, 2.15, 2.675, np.nan], [5.5, 4.75, 4.0, 4.75, 5.5]])
    expected_s_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
    src_crs = target_crs = ccrs.PlateCarree()
    x_grid, y_grid, u_grid, v_grid, s_grid = vec_trans.vector_scalar_to_grid(src_crs, target_crs, (5, 3), self.x, self.y, self.u, self.v, self.s)
    assert_array_equal(x_grid, expected_x_grid)
    assert_array_equal(y_grid, expected_y_grid)
    assert_array_almost_equal(u_grid, expected_u_grid)
    assert_array_almost_equal(v_grid, expected_v_grid)
    assert_array_almost_equal(s_grid, expected_s_grid)