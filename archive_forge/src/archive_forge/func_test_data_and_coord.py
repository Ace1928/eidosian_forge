import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_data_and_coord(self):
    c_data, c_lons = add_cyclic_point(self.data2d, coord=self.lons)
    r_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
    r_lons = np.concatenate((self.lons, np.array([360])))
    assert_array_equal(c_data, r_data)
    assert_array_equal(c_lons, r_lons)