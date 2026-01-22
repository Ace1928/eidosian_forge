import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
def test_line_inside():
    """Line inside"""
    geom1 = load_wkt('LINESTRING (15 15, 16 15)')
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == 'LINESTRING (15 15, 16 15)'