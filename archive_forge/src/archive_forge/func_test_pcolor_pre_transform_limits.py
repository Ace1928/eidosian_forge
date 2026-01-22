import copy
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
import pytest
from matplotlib import scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_pcolor_pre_transform_limits():
    ax = plt.axes()
    xs, ys = np.meshgrid(np.linspace(15, 20, 15), np.linspace(12.4, 12.5, 20))
    ax.pcolor(xs, ys, np.log(xs * ys)[:-1, :-1], transform=mtransforms.Affine2D().scale(0.1) + ax.transData)
    expected = np.array([[1.5, 1.24], [2.0, 1.25]])
    assert_almost_equal(expected, ax.dataLim.get_points())