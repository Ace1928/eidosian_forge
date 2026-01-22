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
def test_secondary_resize():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(2, 11), np.arange(2, 11))

    def invert(x):
        with np.errstate(divide='ignore'):
            return 1 / x
    ax.secondary_xaxis('top', functions=(invert, invert))
    fig.canvas.draw()
    fig.set_size_inches((7, 4))
    assert_allclose(ax.get_position().extents, [0.125, 0.1, 0.9, 0.9])