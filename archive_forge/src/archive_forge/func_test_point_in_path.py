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
def test_point_in_path():
    path = Path._create_closed([(0, 0), (0, 1), (1, 1), (1, 0)])
    points = [(0.5, 0.5), (1.5, 0.5)]
    ret = path.contains_points(points)
    assert ret.dtype == 'bool'
    np.testing.assert_equal(ret, [True, False])