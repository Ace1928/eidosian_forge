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
@pytest.mark.parametrize('use_clabeltext, contour_zorder, clabel_zorder', [(True, 123, 1234), (False, 123, 1234), (True, 123, None), (False, 123, None)])
def test_clabel_zorder(use_clabeltext, contour_zorder, clabel_zorder):
    x, y = np.meshgrid(np.arange(0, 10), np.arange(0, 10))
    z = np.max(np.dstack([abs(x), abs(y)]), 2)
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    cs = ax1.contour(x, y, z, zorder=contour_zorder)
    cs_filled = ax2.contourf(x, y, z, zorder=contour_zorder)
    clabels1 = cs.clabel(zorder=clabel_zorder, use_clabeltext=use_clabeltext)
    clabels2 = cs_filled.clabel(zorder=clabel_zorder, use_clabeltext=use_clabeltext)
    if clabel_zorder is None:
        expected_clabel_zorder = 2 + contour_zorder
    else:
        expected_clabel_zorder = clabel_zorder
    for clabel in clabels1:
        assert clabel.get_zorder() == expected_clabel_zorder
    for clabel in clabels2:
        assert clabel.get_zorder() == expected_clabel_zorder