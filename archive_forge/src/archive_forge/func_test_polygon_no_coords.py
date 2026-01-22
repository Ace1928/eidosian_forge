import pytest
from shapely.geometry import MultiLineString, Polygon, shape
from shapely.geometry.geo import _is_coordinates_empty
@pytest.mark.parametrize('geom', [{'type': 'Polygon', 'coordinates': None}, {'type': 'Polygon', 'coordinates': []}])
def test_polygon_no_coords(geom):
    assert shape(geom) == Polygon()