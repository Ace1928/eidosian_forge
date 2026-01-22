from math import pi
import pytest
from shapely.geometry import Point
from shapely.wkt import dump, dumps, load, loads
def test_dumps_loads_null_geometry(empty_geometry):
    assert dumps(empty_geometry) == 'POINT EMPTY'
    assert loads(dumps(empty_geometry)).equals(empty_geometry)