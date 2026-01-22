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
@image_comparison(['hist_stacked_bar'])
def test_hist_stacked_bar():
    d = [[100, 100, 100, 100, 200, 320, 450, 80, 20, 600, 310, 800], [20, 23, 50, 11, 100, 420], [120, 120, 120, 140, 140, 150, 180], [60, 60, 60, 60, 300, 300, 5, 5, 5, 5, 10, 300], [555, 555, 555, 30, 30, 30, 30, 30, 100, 100, 100, 100, 30, 30], [30, 30, 30, 30, 400, 400, 400, 400, 400, 400, 400, 400]]
    colors = [(0.5759849696758961, 1.0, 0.0), (0.0, 1.0, 0.350624650815206), (0.0, 1.0, 0.6549834156005998), (0.0, 0.6569064625276622, 1.0), (0.28302699607823545, 0.0, 1.0), (0.6849123462299822, 0.0, 1.0)]
    labels = ['green', 'orange', ' yellow', 'magenta', 'black']
    fig, ax = plt.subplots()
    ax.hist(d, bins=10, histtype='barstacked', align='mid', color=colors, label=labels)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncols=1)