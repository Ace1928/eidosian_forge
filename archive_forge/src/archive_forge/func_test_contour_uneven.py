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
@image_comparison(baseline_images=['contour_uneven'], extensions=['png'], remove_text=True, style='mpl20')
def test_contour_uneven(split_collections):
    plt.rcParams['pcolormesh.snap'] = False
    z = np.arange(24).reshape(4, 6)
    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    cs = ax.contourf(z, levels=[2, 4, 6, 10, 20])
    fig.colorbar(cs, ax=ax, spacing='proportional')
    ax = axs[1]
    cs = ax.contourf(z, levels=[2, 4, 6, 10, 20])
    fig.colorbar(cs, ax=ax, spacing='uniform')
    _maybe_split_collections(split_collections)