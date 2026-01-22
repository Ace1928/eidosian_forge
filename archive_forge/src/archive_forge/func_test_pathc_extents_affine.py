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
def test_pathc_extents_affine(self):
    ax = plt.axes()
    offset = mtransforms.Affine2D().translate(10, 10)
    pth = Path([[0, 0], [0, 10], [10, 10], [10, 0]])
    patch = mpatches.PathPatch(pth, transform=offset + ax.transData)
    ax.add_patch(patch)
    expected_data_lim = np.array([[0.0, 0.0], [10.0, 10.0]]) + 10
    assert_array_almost_equal(ax.dataLim.get_points(), expected_data_lim)