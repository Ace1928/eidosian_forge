import numpy as np
import pytest
from numpy.testing import assert_allclose
import shapely
from shapely import MultiLineString, MultiPoint, MultiPolygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_include_z_default():
    _, coords, _ = shapely.to_ragged_array([line_string, line_string_z])
    assert coords.shape[1] == 3
    _, coords, _ = shapely.to_ragged_array([empty_line_string])
    assert coords.shape[1] == 2
    _, coords, _ = shapely.to_ragged_array([empty_line_string_z])
    assert coords.shape[1] == 2
    _, coords, _ = shapely.to_ragged_array(shapely.from_wkt(['MULTIPOLYGON Z EMPTY']))
    assert coords.shape[1] == 2