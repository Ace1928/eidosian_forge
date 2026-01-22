import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cartopy.util import add_cyclic, add_cyclic_point, has_cyclic
def test_invalid_coord_size(self):
    with pytest.raises(ValueError):
        c_data, c_lons = add_cyclic_point(self.data2d, coord=self.lons[:-1])