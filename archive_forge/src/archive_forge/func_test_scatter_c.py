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
@pytest.mark.parametrize('c_case, re_key', params_test_scatter_c)
def test_scatter_c(self, c_case, re_key):

    def get_next_color():
        return 'blue'
    xsize = 4
    REGEXP = {'shape': "^'c' argument has [0-9]+ elements", 'conversion': "^'c' argument must be a color"}
    assert_context = pytest.raises(ValueError, match=REGEXP[re_key]) if re_key is not None else pytest.warns(match='argument looks like a single numeric RGB') if isinstance(c_case, list) and len(c_case) == 3 else contextlib.nullcontext()
    with assert_context:
        mpl.axes.Axes._parse_scatter_color_args(c=c_case, edgecolors='black', kwargs={}, xsize=xsize, get_next_color_func=get_next_color)