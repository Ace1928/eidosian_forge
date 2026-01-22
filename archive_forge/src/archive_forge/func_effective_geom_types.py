import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def effective_geom_types(geom):
    if hasattr(geom, 'geoms') and (not geom.is_empty):
        gts = set()
        for geom in geom.geoms:
            gts |= effective_geom_types(geom)
        return gts
    return {geom.geom_type.lstrip('Multi').replace('LinearRing', 'LineString')}