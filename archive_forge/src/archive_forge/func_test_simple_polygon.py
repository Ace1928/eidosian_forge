import math
import pytest
from shapely.geos import geos_version
from shapely.wkt import loads as load_wkt
@requires_geos_36
def test_simple_polygon():
    poly = load_wkt('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))')
    assert poly.minimum_clearance == 1.0