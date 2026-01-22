import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
def test_transform_correct_coordinate_dimension():
    geom = line_string_z
    assert shapely.get_coordinate_dimension(geom) == 3
    new_geom = transform(geom, lambda x: x + 1, include_z=False)
    assert shapely.get_coordinate_dimension(new_geom) == 2