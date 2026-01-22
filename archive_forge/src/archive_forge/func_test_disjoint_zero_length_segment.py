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
def test_disjoint_zero_length_segment():
    this_path = Path(np.array([[824.85064295, 2056.26489203], [861.69033931, 2041.00539016], [868.57864109, 2057.63522175], [831.73894473, 2072.89472361], [824.85064295, 2056.26489203]]), np.array([1, 2, 2, 2, 79], dtype=Path.code_type))
    outline_path = Path(np.array([[859.91051028, 2165.38461538], [859.06772495, 2149.30331334], [859.06772495, 2181.46591743], [859.91051028, 2165.38461538], [859.91051028, 2165.38461538]]), np.array([1, 2, 2, 2, 2], dtype=Path.code_type))
    assert not outline_path.intersects_path(this_path)
    assert not this_path.intersects_path(outline_path)