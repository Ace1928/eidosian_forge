import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_masked_data(self):
    """Test masked data"""
    new_data = ma.masked_less(self.data2d, 3)
    c_data = add_cyclic(new_data)
    r_data = ma.concatenate((self.data2d, self.data2d[:, :1]), axis=1)
    assert_array_equal(c_data, r_data)
    assert ma.is_masked(c_data)