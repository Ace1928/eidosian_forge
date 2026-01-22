import numpy as np
import pytest
from numpy.testing import assert_allclose
import shapely
from shapely import MultiLineString, MultiPoint, MultiPolygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geom', all_types)
def test_include_z(geom):
    _, coords, _ = shapely.to_ragged_array([geom, geom], include_z=True)
    assert np.isnan(coords[:, 2]).all()