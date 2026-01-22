import numpy as np
import pytest
import shapely
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_normalize_false():
    with pytest.raises(AssertionError):
        assert_geometries_equal(line_string_reversed, line_string, normalize=False)