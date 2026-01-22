import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_data_and_x_y_1d(self):
    """Test data and x and y 1d"""
    c_data, c_lons, c_lats = add_cyclic(self.data2d, x=self.lons, y=self.lats)
    assert_array_equal(c_data, self.c_data2d)
    assert_array_equal(c_lons, self.c_lons)
    assert_array_equal(c_lats, self.c_lats)