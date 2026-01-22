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
def test_hexbin_mincnt_behavior_upon_C_parameter(fig_test, fig_ref):
    datapoints = [(0, 0), (0, 0), (6, 0), (0, 6)]
    X, Y = zip(*datapoints)
    C = [1] * len(X)
    extent = [-10.0, 10, -10.0, 10]
    gridsize = (7, 7)
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()
    ax_ref.hexbin(X, Y, extent=extent, gridsize=gridsize, mincnt=1)
    ax_ref.set_facecolor('green')
    ax_test.hexbin(X, Y, C=[1] * len(X), reduce_C_function=lambda v: sum(v), mincnt=1, extent=extent, gridsize=gridsize)
    ax_test.set_facecolor('green')