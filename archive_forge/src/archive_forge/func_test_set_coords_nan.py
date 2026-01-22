import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
import shapely
from shapely import count_coordinates, get_coordinates, set_coordinates, transform
from shapely.tests.common import (
def test_set_coords_nan():
    geoms = np.array([point])
    coords = np.array([[np.nan, np.inf]])
    new_geoms = set_coordinates(geoms, coords)
    assert_equal(coords, get_coordinates(new_geoms))