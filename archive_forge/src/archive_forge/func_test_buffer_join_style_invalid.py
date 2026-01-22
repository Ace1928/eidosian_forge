import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_buffer_join_style_invalid():
    with pytest.raises(ValueError, match="'invalid' is not a valid option"):
        shapely.buffer(point, 1, join_style='invalid')