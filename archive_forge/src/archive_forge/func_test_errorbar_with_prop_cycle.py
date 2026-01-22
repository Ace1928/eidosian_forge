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
def test_errorbar_with_prop_cycle(fig_test, fig_ref):
    ax = fig_ref.subplots()
    ax.errorbar(x=[2, 4, 10], y=[0, 1, 2], yerr=0.5, ls='--', marker='s', mfc='k')
    ax.errorbar(x=[2, 4, 10], y=[2, 3, 4], yerr=0.5, color='tab:green', ls=':', marker='s', mfc='y')
    ax.errorbar(x=[2, 4, 10], y=[4, 5, 6], yerr=0.5, fmt='tab:blue', ls='-.', marker='o', mfc='c')
    ax.set_xlim(1, 11)
    _cycle = cycler(ls=['--', ':', '-.'], marker=['s', 's', 'o'], mfc=['k', 'y', 'c'], color=['b', 'g', 'r'])
    plt.rc('axes', prop_cycle=_cycle)
    ax = fig_test.subplots()
    ax.errorbar(x=[2, 4, 10], y=[0, 1, 2], yerr=0.5)
    ax.errorbar(x=[2, 4, 10], y=[2, 3, 4], yerr=0.5, color='tab:green')
    ax.errorbar(x=[2, 4, 10], y=[4, 5, 6], yerr=0.5, fmt='tab:blue')
    ax.set_xlim(1, 11)