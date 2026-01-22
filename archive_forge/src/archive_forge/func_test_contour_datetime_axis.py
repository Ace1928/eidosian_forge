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
@image_comparison(['contour_datetime_axis.png'], style='mpl20')
def test_contour_datetime_axis(split_collections):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, top=0.98, bottom=0.15)
    base = datetime.datetime(2013, 1, 1)
    x = np.array([base + datetime.timedelta(days=d) for d in range(20)])
    y = np.arange(20)
    z1, z2 = np.meshgrid(np.arange(20), np.arange(20))
    z = z1 * z2
    plt.subplot(221)
    plt.contour(x, y, z)
    plt.subplot(222)
    plt.contourf(x, y, z)
    x = np.repeat(x[np.newaxis], 20, axis=0)
    y = np.repeat(y[:, np.newaxis], 20, axis=1)
    plt.subplot(223)
    plt.contour(x, y, z)
    plt.subplot(224)
    plt.contourf(x, y, z)
    for ax in fig.get_axes():
        for label in ax.get_xticklabels():
            label.set_ha('right')
            label.set_rotation(30)
    _maybe_split_collections(split_collections)