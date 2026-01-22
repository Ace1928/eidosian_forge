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
@image_comparison(['hist_step_filled.png'], remove_text=True)
def test_hist_step_filled():
    np.random.seed(0)
    x = np.random.randn(1000, 3)
    n_bins = 10
    kwargs = [{'fill': True}, {'fill': False}, {'fill': None}, {}] * 2
    types = ['step'] * 4 + ['stepfilled'] * 4
    fig, axs = plt.subplots(nrows=2, ncols=4)
    for kg, _type, ax in zip(kwargs, types, axs.flat):
        ax.hist(x, n_bins, histtype=_type, stacked=True, **kg)
        ax.set_title(f'{kg}/{_type}')
        ax.set_ylim(bottom=-50)
    patches = axs[0, 0].patches
    assert all((p.get_facecolor() == p.get_edgecolor() for p in patches))