import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygonize_full_array():
    lines = [LineString([(0, 0), (1, 1)]), LineString([(0, 0), (0, 1)]), LineString([(0, 1), (1, 1)])]
    expected = GeometryCollection([Polygon([(1, 1), (0, 0), (0, 1), (1, 1)])])
    result = shapely.polygonize_full(np.array(lines))
    assert len(result) == 4
    assert all((isinstance(geom, shapely.Geometry) for geom in result))
    assert result[0] == expected
    assert all((geom == GeometryCollection() for geom in result[1:]))
    result = shapely.polygonize_full(np.array([lines]))
    assert len(result) == 4
    assert all((isinstance(geom, np.ndarray) for geom in result))
    assert all((geom.shape == (1,) for geom in result))
    assert result[0][0] == expected
    assert all((geom[0] == GeometryCollection() for geom in result[1:]))
    arr = np.array([lines, lines])
    assert arr.shape == (2, 3)
    result = shapely.polygonize_full(arr)
    assert len(result) == 4
    assert all((isinstance(arr, np.ndarray) for arr in result))
    assert all((arr.shape == (2,) for arr in result))
    assert result[0][0] == expected
    assert result[0][1] == expected
    assert all((g == GeometryCollection() for geom in result[1:] for g in geom))
    arr = np.array([[lines, lines], [lines, lines], [lines, lines]])
    assert arr.shape == (3, 2, 3)
    result = shapely.polygonize_full(arr)
    assert len(result) == 4
    assert all((isinstance(arr, np.ndarray) for arr in result))
    assert all((arr.shape == (3, 2) for arr in result))
    for res in result[0].flatten():
        assert res == expected
    for arr in result[1:]:
        for res in arr.flatten():
            assert res == GeometryCollection()