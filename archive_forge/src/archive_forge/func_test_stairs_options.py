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
@image_comparison(['test_stairs_options.png'], remove_text=True)
def test_stairs_options():
    x, y = (np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4]).astype(float))
    yn = y.copy()
    yn[1] = np.nan
    fig, ax = plt.subplots()
    ax.stairs(y * 3, x, color='green', fill=True, label='A')
    ax.stairs(y, x * 3 - 3, color='red', fill=True, orientation='horizontal', label='B')
    ax.stairs(yn, x, color='orange', ls='--', lw=2, label='C')
    ax.stairs(yn / 3, x * 3 - 2, ls='--', lw=2, baseline=0.5, orientation='horizontal', label='D')
    ax.stairs(y[::-1] * 3 + 13, x - 1, color='red', ls='--', lw=2, baseline=None, label='E')
    ax.stairs(y[::-1] * 3 + 14, x, baseline=26, color='purple', ls='--', lw=2, label='F')
    ax.stairs(yn[::-1] * 3 + 15, x + 1, baseline=np.linspace(27, 25, len(y)), color='blue', ls='--', label='G', fill=True)
    ax.stairs(y[:-1][::-1] * 2 + 11, x[:-1] + 0.5, color='black', ls='--', lw=2, baseline=12, hatch='//', label='H')
    ax.legend(loc=0)