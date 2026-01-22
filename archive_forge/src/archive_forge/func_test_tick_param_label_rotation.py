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
def test_tick_param_label_rotation():
    fix, (ax, ax2) = plt.subplots(1, 2)
    ax.plot([0, 1], [0, 1])
    ax2.plot([0, 1], [0, 1])
    ax.xaxis.set_tick_params(which='both', rotation=75)
    ax.yaxis.set_tick_params(which='both', rotation=90)
    for text in ax.get_xticklabels(which='both'):
        assert text.get_rotation() == 75
    for text in ax.get_yticklabels(which='both'):
        assert text.get_rotation() == 90
    ax2.tick_params(axis='x', labelrotation=53)
    ax2.tick_params(axis='y', rotation=35)
    for text in ax2.get_xticklabels(which='major'):
        assert text.get_rotation() == 53
    for text in ax2.get_yticklabels(which='major'):
        assert text.get_rotation() == 35