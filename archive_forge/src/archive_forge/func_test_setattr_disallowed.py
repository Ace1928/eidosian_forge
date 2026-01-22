import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.skipif(platform.python_implementation() == 'PyPy', reason="Setting custom attributes doesn't fail on PyPy")
@pytest.mark.parametrize('geom', geometries_all_types)
def test_setattr_disallowed(geom):
    with pytest.raises(AttributeError):
        geom.name = 'test'