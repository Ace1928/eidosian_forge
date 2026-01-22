import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
@pytest.mark.parametrize('geom_1, geom_2', [[polygon, non_polygon] for non_polygon in non_polygon_types] + [[non_polygon_1, non_polygon_2] for non_polygon_1 in non_polygon_types for non_polygon_2 in non_polygon_types])
def test_coverage_union_non_polygon_inputs(geom_1, geom_2):
    if shapely.geos_version >= (3, 12, 0):

        def effective_geom_types(geom):
            if hasattr(geom, 'geoms') and (not geom.is_empty):
                gts = set()
                for geom in geom.geoms:
                    gts |= effective_geom_types(geom)
                return gts
            return {geom.geom_type.lstrip('Multi').replace('LinearRing', 'LineString')}
        geom_types_1 = effective_geom_types(geom_1)
        geom_types_2 = effective_geom_types(geom_2)
        if len(geom_types_1) == 1 and geom_types_1 == geom_types_2:
            with ignore_invalid():
                result = shapely.coverage_union(geom_1, geom_2)
            assert geom_types_1 == effective_geom_types(result)
        else:
            with pytest.raises(shapely.GEOSException, match='Overlay input is mixed-dimension'):
                shapely.coverage_union(geom_1, geom_2)
    else:
        with pytest.raises(shapely.GEOSException, match='Unhandled geometry type in CoverageUnion.'):
            shapely.coverage_union(geom_1, geom_2)