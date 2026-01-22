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
@image_comparison(['imshow_clip'], style='mpl20', tol=1.24 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0)
def test_imshow_clip():
    matplotlib.rcParams['image.interpolation'] = 'nearest'
    N = 100
    x, y = np.indices((N, N))
    x -= N // 2
    y -= N // 2
    r = np.sqrt(x ** 2 + y ** 2 - x * y)
    fig, ax = plt.subplots()
    c = ax.contour(r, [N / 4])
    clip_path = mtransforms.TransformedPath(c.get_paths()[0], c.get_transform())
    ax.imshow(r, clip_path=clip_path)