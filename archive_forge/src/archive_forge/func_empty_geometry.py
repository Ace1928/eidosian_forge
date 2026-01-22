from math import pi
import pytest
from shapely.geometry import Point
from shapely.wkt import dump, dumps, load, loads
@pytest.fixture(scope='module')
def empty_geometry():
    return Point()