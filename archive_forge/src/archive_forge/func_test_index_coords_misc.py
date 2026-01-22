import numpy as np
import pytest
from shapely import LineString
def test_index_coords_misc(self):
    g = LineString()
    with pytest.raises(IndexError):
        g.coords[0]
    with pytest.raises(TypeError):
        g.coords[0.0]