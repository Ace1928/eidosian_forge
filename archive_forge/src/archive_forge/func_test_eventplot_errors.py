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
@pytest.mark.parametrize('err, args, kwargs, match', ((ValueError, [[1]], {'lineoffsets': []}, 'lineoffsets cannot be empty'), (ValueError, [[1]], {'linelengths': []}, 'linelengths cannot be empty'), (ValueError, [[1]], {'linewidths': []}, 'linewidths cannot be empty'), (ValueError, [[1]], {'linestyles': []}, 'linestyles cannot be empty'), (ValueError, [[1]], {'alpha': []}, 'alpha cannot be empty'), (ValueError, [1], {}, 'positions must be one-dimensional'), (ValueError, [[1]], {'lineoffsets': [1, 2]}, 'lineoffsets and positions are unequal sized sequences'), (ValueError, [[1]], {'linelengths': [1, 2]}, 'linelengths and positions are unequal sized sequences'), (ValueError, [[1]], {'linewidths': [1, 2]}, 'linewidths and positions are unequal sized sequences'), (ValueError, [[1]], {'linestyles': [1, 2]}, 'linestyles and positions are unequal sized sequences'), (ValueError, [[1]], {'alpha': [1, 2]}, 'alpha and positions are unequal sized sequences'), (ValueError, [[1]], {'colors': [1, 2]}, 'colors and positions are unequal sized sequences')))
def test_eventplot_errors(err, args, kwargs, match):
    with pytest.raises(err, match=match):
        plt.eventplot(*args, **kwargs)