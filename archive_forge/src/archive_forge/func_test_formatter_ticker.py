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
@image_comparison(['formatter_ticker_001', 'formatter_ticker_002', 'formatter_ticker_003', 'formatter_ticker_004', 'formatter_ticker_005'])
def test_formatter_ticker():
    import matplotlib.testing.jpl_units as units
    units.register()
    matplotlib.rcParams['lines.markeredgewidth'] = 30
    xdata = [x * units.sec for x in range(10)]
    ydata1 = [(1.5 * y - 0.5) * units.km for y in range(10)]
    ydata2 = [(1.75 * y - 1.0) * units.km for y in range(10)]
    ax = plt.figure().subplots()
    ax.set_xlabel('x-label 001')
    ax = plt.figure().subplots()
    ax.set_xlabel('x-label 001')
    ax.plot(xdata, ydata1, color='blue', xunits='sec')
    ax = plt.figure().subplots()
    ax.set_xlabel('x-label 001')
    ax.plot(xdata, ydata1, color='blue', xunits='sec')
    ax.set_xlabel('x-label 003')
    ax = plt.figure().subplots()
    ax.plot(xdata, ydata1, color='blue', xunits='sec')
    ax.plot(xdata, ydata2, color='green', xunits='hour')
    ax.set_xlabel('x-label 004')
    ax = plt.figure().subplots()
    ax.plot(xdata, ydata1, color='blue', xunits='sec')
    ax.plot(xdata, ydata2, color='green', xunits='hour')
    ax.set_xlabel('x-label 005')
    ax.autoscale_view()