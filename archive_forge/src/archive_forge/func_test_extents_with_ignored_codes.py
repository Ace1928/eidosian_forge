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
@pytest.mark.parametrize('ignored_code', [Path.CLOSEPOLY, Path.STOP])
def test_extents_with_ignored_codes(ignored_code):
    path = Path([[0, 0], [1, 1], [2, 2]], [Path.MOVETO, Path.MOVETO, ignored_code])
    assert np.all(path.get_extents().extents == (0.0, 0.0, 1.0, 1.0))