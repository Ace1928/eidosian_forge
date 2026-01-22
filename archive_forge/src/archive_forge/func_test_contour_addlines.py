import datetime
import platform
import re
from unittest import mock
import contourpy
import numpy as np
from numpy.testing import (
import matplotlib as mpl
from matplotlib import pyplot as plt, rc_context, ticker
from matplotlib.colors import LogNorm, same_color
import matplotlib.patches as mpatches
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import pytest
@pytest.mark.parametrize('split_collections', [False, True])
@image_comparison(['contour_addlines.png'], remove_text=True, style='mpl20', tol=0.15 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0.03)
def test_contour_addlines(split_collections):
    plt.rcParams['pcolormesh.snap'] = False
    fig, ax = plt.subplots()
    np.random.seed(19680812)
    X = np.random.rand(10, 10) * 10000
    pcm = ax.pcolormesh(X)
    cont = ax.contour(X + 1000)
    cb = fig.colorbar(pcm)
    cb.add_lines(cont)
    assert_array_almost_equal(cb.ax.get_ylim(), [114.3091, 9972.30735], 3)
    _maybe_split_collections(split_collections)