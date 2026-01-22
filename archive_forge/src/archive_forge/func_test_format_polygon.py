import pytest
from shapely import Point, Polygon
from shapely.geos import geos_version
def test_format_polygon():
    poly = Point(0, 0).buffer(10, 2)
    assert f'{poly}' == poly.wkt
    assert format(poly, '') == poly.wkt
    assert format(poly, 'x') == poly.wkb_hex.lower()
    assert format(poly, 'X') == poly.wkb_hex
    if geos_version < (3, 13, 0):
        assert f'<{poly:.2f}>' == '<POLYGON ((10.00 0.00, 7.07 -7.07, 0.00 -10.00, -7.07 -7.07, -10.00 -0.00, -7.07 7.07, -0.00 10.00, 7.07 7.07, 10.00 0.00))>'
    else:
        assert f'<{poly:.2f}>' == '<POLYGON ((10.00 0.00, 7.07 -7.07, 0.00 -10.00, -7.07 -7.07, -10.00 0.00, -7.07 7.07, 0.00 10.00, 7.07 7.07, 10.00 0.00))>'
    if geos_version < (3, 10, 0):
        assert f'{poly:.2G}' == 'POLYGON ((10 0, 7.1 -7.1, 1.6E-14 -10, -7.1 -7.1, -10 -3.2E-14, -7.1 7.1, -4.6E-14 10, 7.1 7.1, 10 0))'
    else:
        assert f'{poly:.2G}' == 'POLYGON ((10 0, 7.07 -7.07, 0 -10, -7.07 -7.07, -10 0, -7.07 7.07, 0 10, 7.07 7.07, 10 0))'
    empty = Polygon()
    assert f'{empty}' == 'POLYGON EMPTY'
    assert format(empty, '') == empty.wkt
    assert format(empty, '.2G') == empty.wkt
    assert format(empty, 'x') == empty.wkb_hex.lower()
    assert format(empty, 'X') == empty.wkb_hex