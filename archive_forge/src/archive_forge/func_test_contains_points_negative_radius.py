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
def test_contains_points_negative_radius():
    path = Path.unit_circle()
    points = [(0.0, 0.0), (1.25, 0.0), (0.9, 0.9)]
    result = path.contains_points(points, radius=-0.5)
    np.testing.assert_equal(result, [True, False, False])