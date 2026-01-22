import math
import pytest
from shapely.geos import geos_version
from shapely.wkt import loads as load_wkt
@requires_geos_36
def test_more_complicated_polygon():
    poly = load_wkt('POLYGON ((20 20, 34 124, 70 140, 130 130, 70 100, 110 70, 170 20, 90 10, 20 20))')
    assert round(poly.minimum_clearance, 6) == 35.777088