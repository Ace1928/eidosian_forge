import numpy as np
import pytest
from shapely import Point
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError
def test_numpy_empty_point_coords():
    pe = Point()
    a = np.asarray(pe.coords)
    assert a.shape == (0, 2)