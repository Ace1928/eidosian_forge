import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_parts_geometry_collection_multi():
    """On the first pass, the individual Multi* geometry objects are returned
    from the collection.  On the second pass, the individual singular geometry
    objects within those are returned.
    """
    geom = shapely.geometrycollections([multi_point, multi_line_string, multi_polygon])
    expected_num_parts = shapely.get_num_geometries(geom)
    expected_parts = shapely.get_geometry(geom, range(0, expected_num_parts))
    parts = shapely.get_parts(geom)
    assert len(parts) == expected_num_parts
    assert_geometries_equal(parts, expected_parts)
    expected_subparts = []
    for g in np.asarray(expected_parts):
        for i in range(0, shapely.get_num_geometries(g)):
            expected_subparts.append(shapely.get_geometry(g, i))
    subparts = shapely.get_parts(parts)
    assert len(subparts) == len(expected_subparts)
    assert_geometries_equal(subparts, expected_subparts)