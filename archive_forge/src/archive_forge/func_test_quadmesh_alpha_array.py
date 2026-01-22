from datetime import datetime
import io
import itertools
import re
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
from matplotlib.collections import (Collection, LineCollection,
from matplotlib.testing.decorators import check_figures_equal, image_comparison
def test_quadmesh_alpha_array(pcfunc):
    x = np.arange(4)
    y = np.arange(4)
    z = np.arange(9).reshape((3, 3))
    alpha = z / z.max()
    alpha_flat = alpha.ravel()
    fig, (ax0, ax1) = plt.subplots(2)
    coll1 = getattr(ax0, pcfunc)(x, y, z, alpha=alpha)
    coll2 = getattr(ax0, pcfunc)(x, y, z)
    coll2.set_alpha(alpha)
    plt.draw()
    assert_array_equal(coll1.get_facecolors()[:, -1], alpha_flat)
    assert_array_equal(coll2.get_facecolors()[:, -1], alpha_flat)
    fig, (ax0, ax1) = plt.subplots(2)
    coll1 = getattr(ax0, pcfunc)(x, y, z, alpha=alpha)
    coll2 = getattr(ax1, pcfunc)(x, y, z)
    coll2.set_alpha(alpha)
    plt.draw()
    assert_array_equal(coll1.get_facecolors()[:, -1], alpha_flat)
    assert_array_equal(coll2.get_facecolors()[:, -1], alpha_flat)