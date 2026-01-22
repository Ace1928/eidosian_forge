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
def test_point_in_path_nan():
    box = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    p = Path(box)
    test = np.array([[np.nan, 0.5]])
    contains = p.contains_points(test)
    assert len(contains) == 1
    assert not contains[0]