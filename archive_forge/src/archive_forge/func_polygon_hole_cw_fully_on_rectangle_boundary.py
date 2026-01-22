import pytest
from shapely.ops import clip_by_rect
from shapely.wkt import dumps as dump_wkt
from shapely.wkt import loads as load_wkt
def polygon_hole_cw_fully_on_rectangle_boundary():
    """Polygon hole (CW) fully on rectangle boundary"""
    geom1 = load_wkt('POLYGON ((0 0, 0 30, 30 30, 30 0, 0 0), (10 10, 10 20, 20 20, 20 10, 10 10))')
    geom2 = clip_by_rect(geom1, 10, 10, 20, 20)
    assert dump_wkt(geom2, rounding_precision=0) == 'GEOMETRYCOLLECTION EMPTY'