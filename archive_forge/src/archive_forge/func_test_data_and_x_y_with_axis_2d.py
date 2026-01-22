import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_data_and_x_y_with_axis_2d(self):
    """Test axis keyword data 4d, x and y 2d"""
    c_data, c_lons, c_lats = add_cyclic(self.data4d, x=self.lon2d, y=self.lat2d, axis=1)
    assert_array_equal(c_data, self.c_data4d)
    assert_array_equal(c_lons, self.c_lon2d)
    assert_array_equal(c_lats, self.c_lat2d)