import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_linestring_array_coercion():
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    arr = np.array(line)
    assert arr.ndim == 0
    assert arr.size == 1
    assert arr.dtype == np.dtype('object')
    assert arr.item() == line