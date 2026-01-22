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
@image_comparison(['test_loglog_nonpos.png'], remove_text=True, style='mpl20')
def test_loglog_nonpos():
    fig, axs = plt.subplots(3, 3)
    x = np.arange(1, 11)
    y = x ** 3
    y[7] = -3.0
    x[4] = -10
    for (mcy, mcx), ax in zip(product(['mask', 'clip', ''], repeat=2), axs.flat):
        if mcx == mcy:
            if mcx:
                ax.loglog(x, y ** 3, lw=2, nonpositive=mcx)
            else:
                ax.loglog(x, y ** 3, lw=2)
        else:
            ax.loglog(x, y ** 3, lw=2)
            if mcx:
                ax.set_xscale('log', nonpositive=mcx)
            if mcy:
                ax.set_yscale('log', nonpositive=mcy)