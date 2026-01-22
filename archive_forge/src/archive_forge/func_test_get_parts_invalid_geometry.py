import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom', ['foo', ['foo'], 42])
def test_get_parts_invalid_geometry(geom):
    with pytest.raises(TypeError, match='One of the arguments is of incorrect type.'):
        shapely.get_parts(geom)