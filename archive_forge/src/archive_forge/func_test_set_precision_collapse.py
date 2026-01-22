import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geometry,mode,expected', [(Polygon([(2, 2), (4, 2), (3.2, 3), (4, 4), (2, 4), (2.8, 3), (2, 2)]), 'valid_output', MultiPolygon([Polygon([(4, 2), (2, 2), (3, 3), (4, 2)]), Polygon([(2, 4), (4, 4), (3, 3), (2, 4)])])), pytest.param(Polygon([(2, 2), (4, 2), (3.2, 3), (4, 4), (2, 4), (2.8, 3), (2, 2)]), 'pointwise', Polygon([(2, 2), (4, 2), (3, 3), (4, 4), (2, 4), (3, 3), (2, 2)]), marks=pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='pointwise does not work pre-GEOS 3.10')), (Polygon([(2, 2), (4, 2), (3.2, 3), (4, 4), (2, 4), (2.8, 3), (2, 2)]), 'keep_collapsed', MultiPolygon([Polygon([(4, 2), (2, 2), (3, 3), (4, 2)]), Polygon([(2, 4), (4, 4), (3, 3), (2, 4)])])), (LineString([(0, 0), (0.1, 0.1)]), 'valid_output', LineString()), pytest.param(LineString([(0, 0), (0.1, 0.1)]), 'pointwise', LineString([(0, 0), (0, 0)]), marks=pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='pointwise does not work pre-GEOS 3.10')), (LineString([(0, 0), (0.1, 0.1)]), 'keep_collapsed', LineString([(0, 0), (0, 0)])), pytest.param(LinearRing([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]), 'valid_output', LinearRing(), marks=pytest.mark.skipif(shapely.geos_version == (3, 10, 0), reason='Segfaults on GEOS 3.10.0')), pytest.param(LinearRing([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]), 'pointwise', LinearRing([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]), marks=pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='pointwise does not work pre-GEOS 3.10')), pytest.param(LinearRing([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]), 'keep_collapsed', LineString([(0, 0), (0, 0), (0, 0)]), marks=pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='this collapsed into an invalid linearring pre-GEOS 3.10')), (Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]), 'valid_output', Polygon()), pytest.param(Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]), 'pointwise', Polygon([(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]), marks=pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='pointwise does not work pre-GEOS 3.10')), (Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, 0)]), 'keep_collapsed', Polygon())])
def test_set_precision_collapse(geometry, mode, expected):
    """Lines and polygons collapse to empty geometries if vertices are too close"""
    actual = shapely.set_precision(geometry, 1, mode=mode)
    if shapely.geos_version < (3, 9, 0):
        assert shapely.to_wkt(shapely.normalize(actual)) == shapely.to_wkt(shapely.normalize(expected))
    else:
        assert_geometries_equal(shapely.force_2d(actual), expected)