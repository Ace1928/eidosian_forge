import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_has_cyclic_2d_full(self):
    """Test detection of cyclic point 2d including y"""
    c_data, c_lons, c_lats = add_cyclic(self.c_data2d, x=self.c_lon2d, y=self.c_lat2d)
    assert_array_equal(c_data, self.c_data2d)
    assert_array_equal(c_lons, self.c_lon2d)
    assert_array_equal(c_lats, self.c_lat2d)