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
def test_quiver_limits():
    ax = plt.axes()
    x, y = (np.arange(8), np.arange(10))
    u = v = np.linspace(0, 10, 80).reshape(10, 8)
    q = plt.quiver(x, y, u, v)
    assert q.get_datalim(ax.transData).bounds == (0.0, 0.0, 7.0, 9.0)
    plt.figure()
    ax = plt.axes()
    x = np.linspace(-5, 10, 20)
    y = np.linspace(-2, 4, 10)
    y, x = np.meshgrid(y, x)
    trans = mtransforms.Affine2D().translate(25, 32) + ax.transData
    plt.quiver(x, y, np.sin(x), np.cos(y), transform=trans)
    assert ax.dataLim.bounds == (20.0, 30.0, 15.0, 6.0)