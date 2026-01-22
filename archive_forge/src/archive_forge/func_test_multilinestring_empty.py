import pytest
from shapely.geometry import MultiLineString, Polygon, shape
from shapely.geometry.geo import _is_coordinates_empty
@pytest.mark.parametrize('geom', [{'type': 'MultiLineString', 'coordinates': []}, {'type': 'MultiLineString', 'coordinates': [[]]}, {'type': 'MultiLineString', 'coordinates': None}])
def test_multilinestring_empty(geom):
    assert shape(geom) == MultiLineString()