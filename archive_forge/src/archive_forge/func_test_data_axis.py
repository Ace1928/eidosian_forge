import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
@pytest.mark.parametrize('lon, clon, axis', [(lons, c_lons, 0), (lon2d, c_lon2d, 1), (ma.masked_inside(lon2d, 100, 200), ma.masked_inside(c_lon2d, 100, 200), 1)])
def test_data_axis(self, lon, clon, axis):
    """Test lon is not cyclic, clon is cyclic, with axis keyword"""
    assert not has_cyclic(lon, axis=axis)
    assert has_cyclic(clon, axis=axis)