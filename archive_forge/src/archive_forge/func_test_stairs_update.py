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
@check_figures_equal(extensions=['png'])
def test_stairs_update(fig_test, fig_ref):
    ylim = (-3, 4)
    test_ax = fig_test.add_subplot()
    h = test_ax.stairs([1, 2, 3])
    test_ax.set_ylim(ylim)
    h.set_data([3, 2, 1])
    h.set_data(edges=np.arange(4) + 2)
    h.set_data([1, 2, 1], np.arange(4) / 2)
    h.set_data([1, 2, 3])
    h.set_data(None, np.arange(4))
    assert np.allclose(h.get_data()[0], np.arange(1, 4))
    assert np.allclose(h.get_data()[1], np.arange(4))
    h.set_data(baseline=-2)
    assert h.get_data().baseline == -2
    ref_ax = fig_ref.add_subplot()
    h = ref_ax.stairs([1, 2, 3], baseline=-2)
    ref_ax.set_ylim(ylim)