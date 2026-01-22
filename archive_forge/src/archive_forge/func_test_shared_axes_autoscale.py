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
def test_shared_axes_autoscale():
    l = np.arange(-80, 90, 40)
    t = np.random.random_sample((l.size, l.size))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    ax1.set_xlim(-1000, 1000)
    ax1.set_ylim(-1000, 1000)
    ax1.contour(l, l, t)
    ax2.contour(l, l, t)
    assert not ax1.get_autoscalex_on() and (not ax2.get_autoscalex_on())
    assert not ax1.get_autoscaley_on() and (not ax2.get_autoscaley_on())
    assert ax1.get_xlim() == ax2.get_xlim() == (-1000, 1000)
    assert ax1.get_ylim() == ax2.get_ylim() == (-1000, 1000)