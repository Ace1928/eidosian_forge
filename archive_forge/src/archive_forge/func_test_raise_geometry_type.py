import numpy as np
import pytest
from numpy.testing import assert_allclose
import shapely
from shapely import MultiLineString, MultiPoint, MultiPolygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geom', all_types_not_supported)
def test_raise_geometry_type(geom):
    with pytest.raises(ValueError):
        shapely.to_ragged_array([geom, geom])