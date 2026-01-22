import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
def test_colorbar_autoticks():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(2, 1)
        x = np.arange(-3.0, 4.001)
        y = np.arange(-4.0, 3.001)
        X, Y = np.meshgrid(x, y)
        Z = X * Y
        Z = Z[:-1, :-1]
        pcm = ax[0].pcolormesh(X, Y, Z)
        cbar = fig.colorbar(pcm, ax=ax[0], extend='both', orientation='vertical')
        pcm = ax[1].pcolormesh(X, Y, Z)
        cbar2 = fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical', shrink=0.4)
        np.testing.assert_almost_equal(cbar.ax.yaxis.get_ticklocs(), np.arange(-15, 16, 5))
        np.testing.assert_almost_equal(cbar2.ax.yaxis.get_ticklocs(), np.arange(-20, 21, 10))