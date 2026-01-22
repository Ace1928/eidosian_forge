import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_3d_axis(self):
    """Test 3d with axis keyword, no keyword name for axis"""
    assert has_cyclic(self.c_lon3d, 1)
    assert not has_cyclic(self.lon3d, 1)