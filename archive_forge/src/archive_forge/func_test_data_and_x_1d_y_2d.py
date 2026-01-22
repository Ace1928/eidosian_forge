import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_data_and_x_1d_y_2d(self):
    """Test data and x 1d and y 2d"""
    c_data, c_lons, c_lats = add_cyclic(self.data2d, x=self.lons, y=self.lat2d)
    assert_array_equal(c_data, self.c_data2d)
    assert_array_equal(c_lons, self.c_lons)
    assert_array_equal(c_lats, self.c_lat2d)