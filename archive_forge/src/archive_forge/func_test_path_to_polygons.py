import re
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from matplotlib import patches
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.backend_bases import MouseEvent
def test_path_to_polygons():
    data = [[10, 10], [20, 20]]
    p = Path(data)
    assert_array_equal(p.to_polygons(width=40, height=40), [])
    assert_array_equal(p.to_polygons(width=40, height=40, closed_only=False), [data])
    assert_array_equal(p.to_polygons(), [])
    assert_array_equal(p.to_polygons(closed_only=False), [data])
    data = [[10, 10], [20, 20], [30, 30]]
    closed_data = [[10, 10], [20, 20], [30, 30], [10, 10]]
    p = Path(data)
    assert_array_equal(p.to_polygons(width=40, height=40), [closed_data])
    assert_array_equal(p.to_polygons(width=40, height=40, closed_only=False), [data])
    assert_array_equal(p.to_polygons(), [closed_data])
    assert_array_equal(p.to_polygons(closed_only=False), [data])