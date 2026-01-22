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
@image_comparison(['arc_ellipse'], remove_text=True)
def test_arc_ellipse():
    xcenter, ycenter = (0.38, 0.52)
    width, height = (0.1, 0.3)
    angle = -30
    theta = np.deg2rad(np.arange(360))
    x = width / 2.0 * np.cos(theta)
    y = height / 2.0 * np.sin(theta)
    rtheta = np.deg2rad(angle)
    R = np.array([[np.cos(rtheta), -np.sin(rtheta)], [np.sin(rtheta), np.cos(rtheta)]])
    x, y = np.dot(R, [x, y])
    x += xcenter
    y += ycenter
    fig = plt.figure()
    ax = fig.add_subplot(211, aspect='auto')
    ax.fill(x, y, alpha=0.2, facecolor='yellow', edgecolor='yellow', linewidth=1, zorder=1)
    e1 = mpatches.Arc((xcenter, ycenter), width, height, angle=angle, linewidth=2, fill=False, zorder=2)
    ax.add_patch(e1)
    ax = fig.add_subplot(212, aspect='equal')
    ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)
    e2 = mpatches.Arc((xcenter, ycenter), width, height, angle=angle, linewidth=2, fill=False, zorder=2)
    ax.add_patch(e2)