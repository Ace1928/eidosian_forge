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
def test_xticks_bad_args():
    ax = plt.figure().add_subplot()
    with pytest.raises(TypeError, match='must be a sequence'):
        ax.set_xticks([2, 9], 3.1)
    with pytest.raises(ValueError, match='must be 1D'):
        plt.xticks(np.arange(4).reshape((-1, 1)))
    with pytest.raises(ValueError, match='must be 1D'):
        plt.xticks(np.arange(4).reshape((1, -1)))
    with pytest.raises(ValueError, match='must be 1D'):
        plt.xticks(np.arange(4).reshape((-1, 1)), labels=range(4))
    with pytest.raises(ValueError, match='must be 1D'):
        plt.xticks(np.arange(4).reshape((1, -1)), labels=range(4))