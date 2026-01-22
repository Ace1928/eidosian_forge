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
@pytest.mark.parametrize('fv, fh, args', [[plt.axvline, plt.axhline, (1,)], [plt.axvspan, plt.axhspan, (1, 1)]])
def test_axline_minmax(fv, fh, args):
    bad_lim = matplotlib.dates.num2date(1)
    with pytest.raises(ValueError, match='ymin must be a single scalar value'):
        fv(*args, ymin=bad_lim, ymax=1)
    with pytest.raises(ValueError, match='ymax must be a single scalar value'):
        fv(*args, ymin=1, ymax=bad_lim)
    with pytest.raises(ValueError, match='xmin must be a single scalar value'):
        fh(*args, xmin=bad_lim, xmax=1)
    with pytest.raises(ValueError, match='xmax must be a single scalar value'):
        fh(*args, xmin=1, xmax=bad_lim)