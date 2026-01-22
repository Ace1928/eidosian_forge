import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (

    This is tested seperately from other set operations as it differs in two ways:
      1. It expects only non-overlapping polygons
      2. It expects GEOS 3.8.0+
    