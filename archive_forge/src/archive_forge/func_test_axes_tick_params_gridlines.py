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
def test_axes_tick_params_gridlines():
    ax = plt.subplot()
    ax.tick_params(grid_color='b', grid_linewidth=5, grid_alpha=0.5, grid_linestyle='dashdot')
    for axis in (ax.xaxis, ax.yaxis):
        assert axis.majorTicks[0].gridline.get_color() == 'b'
        assert axis.majorTicks[0].gridline.get_linewidth() == 5
        assert axis.majorTicks[0].gridline.get_alpha() == 0.5
        assert axis.majorTicks[0].gridline.get_linestyle() == '-.'