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
@image_comparison(['boxplot_rc_parameters'], savefig_kwarg={'dpi': 100}, remove_text=True, tol=1, style='default')
def test_boxplot_rc_parameters():
    np.random.seed(937)
    fig, ax = plt.subplots(3)
    rc_axis0 = {'boxplot.notch': True, 'boxplot.whiskers': [5, 95], 'boxplot.bootstrap': 10000, 'boxplot.flierprops.color': 'b', 'boxplot.flierprops.marker': 'o', 'boxplot.flierprops.markerfacecolor': 'g', 'boxplot.flierprops.markeredgecolor': 'b', 'boxplot.flierprops.markersize': 5, 'boxplot.flierprops.linestyle': '--', 'boxplot.flierprops.linewidth': 2.0, 'boxplot.boxprops.color': 'r', 'boxplot.boxprops.linewidth': 2.0, 'boxplot.boxprops.linestyle': '--', 'boxplot.capprops.color': 'c', 'boxplot.capprops.linewidth': 2.0, 'boxplot.capprops.linestyle': '--', 'boxplot.medianprops.color': 'k', 'boxplot.medianprops.linewidth': 2.0, 'boxplot.medianprops.linestyle': '--'}
    rc_axis1 = {'boxplot.vertical': False, 'boxplot.whiskers': [0, 100], 'boxplot.patchartist': True}
    rc_axis2 = {'boxplot.whiskers': 2.0, 'boxplot.showcaps': False, 'boxplot.showbox': False, 'boxplot.showfliers': False, 'boxplot.showmeans': True, 'boxplot.meanline': True, 'boxplot.meanprops.color': 'c', 'boxplot.meanprops.linewidth': 2.0, 'boxplot.meanprops.linestyle': '--', 'boxplot.whiskerprops.color': 'r', 'boxplot.whiskerprops.linewidth': 2.0, 'boxplot.whiskerprops.linestyle': '-.'}
    dict_list = [rc_axis0, rc_axis1, rc_axis2]
    for axis, rc_axis in zip(ax, dict_list):
        _rc_test_bxp_helper(axis, rc_axis)
    assert matplotlib.patches.PathPatch in [type(t) for t in ax[1].get_children()]