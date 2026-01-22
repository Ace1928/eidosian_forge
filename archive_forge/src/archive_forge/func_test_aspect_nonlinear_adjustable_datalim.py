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
def test_aspect_nonlinear_adjustable_datalim():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot([0.4, 0.6], [0.4, 0.6])
    ax.set(xscale='log', xlim=(1, 100), yscale='logit', ylim=(1 / 101, 1 / 11), aspect=1, adjustable='datalim')
    ax.margins(0)
    ax.apply_aspect()
    assert ax.get_xlim() == pytest.approx([1 * 10 ** (1 / 2), 100 / 10 ** (1 / 2)])
    assert ax.get_ylim() == (1 / 101, 1 / 11)