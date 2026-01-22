import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
def test_type_deprecated():
    geom = Point(1, 1)
    with pytest.warns(ShapelyDeprecationWarning):
        geom_type = geom.type
    assert geom_type == geom.geom_type