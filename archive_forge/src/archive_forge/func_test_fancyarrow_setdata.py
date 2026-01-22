import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
def test_fancyarrow_setdata():
    fig, ax = plt.subplots()
    arrow = ax.arrow(0, 0, 10, 10, head_length=5, head_width=1, width=0.5)
    expected1 = np.array([[13.54, 13.54], [10.35, 9.65], [10.18, 9.82], [0.18, -0.18], [-0.18, 0.18], [9.82, 10.18], [9.65, 10.35], [13.54, 13.54]])
    assert np.allclose(expected1, np.round(arrow.verts, 2))
    expected2 = np.array([[16.71, 16.71], [16.71, 15.29], [16.71, 15.29], [1.71, 0.29], [0.29, 1.71], [15.29, 16.71], [15.29, 16.71], [16.71, 16.71]])
    arrow.set_data(x=1, y=1, dx=15, dy=15, width=2, head_width=2, head_length=1)
    assert np.allclose(expected2, np.round(arrow.verts, 2))