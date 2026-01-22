import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_precision_has_cyclic_no(self):
    """Test precision keyword detecting no cyclic point"""
    new_data = np.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
    new_lons = np.concatenate((self.lons, np.array([360.0 + 0.001])))
    c_data, c_lons = add_cyclic(new_data, x=new_lons, precision=0.0002)
    r_data = np.concatenate((new_data, new_data[:, :1]), axis=1)
    r_lons = np.concatenate((new_lons, np.array([360])))
    assert_array_equal(c_data, r_data)
    assert_array_equal(c_lons, r_lons)