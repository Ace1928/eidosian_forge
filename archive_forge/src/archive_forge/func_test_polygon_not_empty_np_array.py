import pytest
from shapely.geometry import MultiLineString, Polygon, shape
from shapely.geometry.geo import _is_coordinates_empty
def test_polygon_not_empty_np_array():
    np = pytest.importorskip('numpy')
    geom = {'type': 'Polygon', 'coordinates': np.array([[[5, 10], [10, 10], [10, 5]]])}
    obj = shape(geom)
    assert obj == Polygon([(5, 10), (10, 10), (10, 5)])