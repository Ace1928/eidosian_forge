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
@pytest.mark.parametrize('offset', range(-720, 361, 45))
def test_full_arc(offset):
    low = offset
    high = 360 + offset
    path = Path.arc(low, high)
    mins = np.min(path.vertices, axis=0)
    maxs = np.max(path.vertices, axis=0)
    np.testing.assert_allclose(mins, -1)
    np.testing.assert_allclose(maxs, 1)