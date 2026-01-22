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
@image_comparison(['axisbelow.png'], remove_text=True)
def test_axisbelow():
    axs = plt.figure().subplots(ncols=3, sharex=True, sharey=True)
    settings = (False, 'line', True)
    for ax, setting in zip(axs, settings):
        ax.plot((0, 10), (0, 10), lw=10, color='m')
        circ = mpatches.Circle((3, 3), color='r')
        ax.add_patch(circ)
        ax.grid(color='c', linestyle='-', linewidth=3)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        ax.spines[:].set_visible(False)
        ax.set_axisbelow(setting)
        assert ax.get_axisbelow() == setting