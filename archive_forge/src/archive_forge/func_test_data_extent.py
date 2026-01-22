import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import cartopy.crs as ccrs
import cartopy.vector_transform as vec_trans
def test_data_extent(self):
    expected_x_grid = np.array([[-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0], [-10.0, -5.0, 0.0, 5.0, 10.0]])
    expected_y_grid = np.array([[5.0, 5.0, 5.0, 5.0, 5.0], [7.5, 7.5, 7.5, 7.5, 7.5], [10.0, 10.0, 10.0, 10.0, 10]])
    expected_s_grid = np.array([[np.nan, 2.0, 3.0, 2.0, np.nan], [np.nan, 2.5, 3.5, 2.5, np.nan], [2.0, 3.0, 4.0, 3.0, 2.0]])
    x_grid, y_grid, s_grid = vec_trans._interpolate_to_grid(5, 3, self.x, self.y, self.s)
    assert_array_equal(x_grid, expected_x_grid)
    assert_array_equal(y_grid, expected_y_grid)
    assert_array_almost_equal(s_grid, expected_s_grid)