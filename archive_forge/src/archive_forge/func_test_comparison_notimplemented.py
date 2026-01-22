import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
@pytest.mark.parametrize('geom', all_types)
def test_comparison_notimplemented(geom):
    assert geom.__eq__(1) is NotImplemented
    arr = np.array([geom, geom], dtype=object)
    result = arr == geom
    assert isinstance(result, np.ndarray)
    assert result.all()
    result = geom == arr
    assert isinstance(result, np.ndarray)
    assert result.all()
    result = arr != geom
    assert isinstance(result, np.ndarray)
    assert not result.any()
    result = geom != arr
    assert isinstance(result, np.ndarray)
    assert not result.any()