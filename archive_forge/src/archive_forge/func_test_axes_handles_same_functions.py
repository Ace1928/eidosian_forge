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
@check_figures_equal(extensions=['png'])
def test_axes_handles_same_functions(fig_ref, fig_test):
    for nn, fig in enumerate([fig_ref, fig_test]):
        ax = fig.add_subplot()
        pc = ax.pcolormesh(np.ones(300).reshape(10, 30))
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb = fig.colorbar(pc, cax=cax)
        if nn == 0:
            caxx = cax
        else:
            caxx = cb.ax
        caxx.set_yticks(np.arange(0, 20))
        caxx.set_yscale('log')
        caxx.set_position([0.92, 0.1, 0.02, 0.7])