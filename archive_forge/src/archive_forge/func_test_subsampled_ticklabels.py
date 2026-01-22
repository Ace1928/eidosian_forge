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
def test_subsampled_ticklabels():
    fig, ax = plt.subplots()
    ax.plot(np.arange(10))
    ax.xaxis.set_ticks(np.arange(10) + 0.1)
    ax.locator_params(nbins=5)
    ax.xaxis.set_ticklabels([c for c in 'bcdefghijk'])
    plt.draw()
    labels = [t.get_text() for t in ax.xaxis.get_ticklabels()]
    assert labels == ['b', 'd', 'f', 'h', 'j']