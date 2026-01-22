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
@check_figures_equal()
def test_errorbar_every(fig_test, fig_ref):
    x = np.linspace(0, 1, 15)
    y = x * (1 - x)
    yerr = y / 6
    ax_ref = fig_ref.subplots()
    ax_test = fig_test.subplots()
    for color, shift in zip('rgbk', [0, 0, 2, 7]):
        y += 0.02
        ax_test.errorbar(x, y, yerr, errorevery=(shift, 4), capsize=4, c=color)
        ax_ref.plot(x, y, c=color, zorder=2.1)
        ax_ref.errorbar(x[shift::4], y[shift::4], yerr[shift::4], capsize=4, c=color, fmt='none')
    ax_test.errorbar(x, y + 0.1, yerr, markevery=(1, 4), capsize=4, fmt='o')
    ax_ref.plot(x[1::4], y[1::4] + 0.1, 'o', zorder=2.1)
    ax_ref.errorbar(x, y + 0.1, yerr, capsize=4, fmt='none')
    ax_test.errorbar(x, y + 0.2, yerr, errorevery=slice(2, None, 3), markevery=slice(2, None, 3), capsize=4, c='C0', fmt='o')
    ax_ref.plot(x[2::3], y[2::3] + 0.2, 'o', c='C0', zorder=2.1)
    ax_ref.errorbar(x[2::3], y[2::3] + 0.2, yerr[2::3], capsize=4, c='C0', fmt='none')
    ax_test.errorbar(x, y + 0.2, yerr, errorevery=[False, True, False] * 5, markevery=[False, True, False] * 5, capsize=4, c='C1', fmt='o')
    ax_ref.plot(x[1::3], y[1::3] + 0.2, 'o', c='C1', zorder=2.1)
    ax_ref.errorbar(x[1::3], y[1::3] + 0.2, yerr[1::3], capsize=4, c='C1', fmt='none')