import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.parametrize('geom', geometries_all_types)
def test_weakrefable(geom):
    _ = weakref.ref(geom)