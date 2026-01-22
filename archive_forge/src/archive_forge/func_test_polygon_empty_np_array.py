import pytest
from shapely.geometry import MultiLineString, Polygon, shape
from shapely.geometry.geo import _is_coordinates_empty
def test_polygon_empty_np_array():
    np = pytest.importorskip('numpy')
    geom = {'type': 'Polygon', 'coordinates': np.array([])}
    assert shape(geom) == Polygon()