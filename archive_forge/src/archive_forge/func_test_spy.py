import contextlib
from collections import namedtuple
import datetime
from decimal import Decimal
from functools import partial
import inspect
import io
from itertools import product
import platform
from types import SimpleNamespace
import dateutil.tz
import numpy as np
from numpy import ma
from cycler import cycler
import pytest
import matplotlib
import matplotlib as mpl
from matplotlib import rc_context, patheffects
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.font_manager as mfont_manager
import matplotlib.markers as mmarkers
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import mpl_toolkits.axisartist as AA  # type: ignore
from numpy.testing import (
from matplotlib.testing.decorators import (
@check_figures_equal(extensions=['png'])
def test_spy(fig_test, fig_ref):
    np.random.seed(19680801)
    a = np.ones(32 * 32)
    a[:16 * 32] = 0
    np.random.shuffle(a)
    a = a.reshape((32, 32))
    axs_test = fig_test.subplots(2)
    axs_test[0].spy(a)
    axs_test[1].spy(a, marker='.', origin='lower')
    axs_ref = fig_ref.subplots(2)
    axs_ref[0].imshow(a, cmap='gray_r', interpolation='nearest')
    axs_ref[0].xaxis.tick_top()
    axs_ref[1].plot(*np.nonzero(a)[::-1], '.', markersize=10)
    axs_ref[1].set(aspect=1, xlim=axs_ref[0].get_xlim(), ylim=axs_ref[0].get_ylim()[::-1])
    for ax in axs_ref:
        ax.xaxis.set_ticks_position('both')