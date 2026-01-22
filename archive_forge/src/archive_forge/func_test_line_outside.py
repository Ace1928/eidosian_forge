import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
def test_line_outside():
    """Line outside"""
    geom1 = load_wkt('LINESTRING (0 0, -5 5)')
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == 'GEOMETRYCOLLECTION EMPTY'