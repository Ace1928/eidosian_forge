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
def test_small_autoscale():
    verts = np.array([[-5.45, 0.0], [-5.45, 0.0], [-5.29, 0.0], [-5.29, 0.0], [-5.13, 0.0], [-5.13, 0.0], [-4.97, 0.0], [-4.97, 0.0], [-4.81, 0.0], [-4.81, 0.0], [-4.65, 0.0], [-4.65, 0.0], [-4.49, 0.0], [-4.49, 0.0], [-4.33, 0.0], [-4.33, 0.0], [-4.17, 0.0], [-4.17, 0.0], [-4.01, 0.0], [-4.01, 0.0], [-3.85, 0.0], [-3.85, 0.0], [-3.69, 0.0], [-3.69, 0.0], [-3.53, 0.0], [-3.53, 0.0], [-3.37, 0.0], [-3.37, 0.0], [-3.21, 0.0], [-3.21, 0.01], [-3.05, 0.01], [-3.05, 0.01], [-2.89, 0.01], [-2.89, 0.01], [-2.73, 0.01], [-2.73, 0.02], [-2.57, 0.02], [-2.57, 0.04], [-2.41, 0.04], [-2.41, 0.04], [-2.25, 0.04], [-2.25, 0.06], [-2.09, 0.06], [-2.09, 0.08], [-1.93, 0.08], [-1.93, 0.1], [-1.77, 0.1], [-1.77, 0.12], [-1.61, 0.12], [-1.61, 0.14], [-1.45, 0.14], [-1.45, 0.17], [-1.3, 0.17], [-1.3, 0.19], [-1.14, 0.19], [-1.14, 0.22], [-0.98, 0.22], [-0.98, 0.25], [-0.82, 0.25], [-0.82, 0.27], [-0.66, 0.27], [-0.66, 0.29], [-0.5, 0.29], [-0.5, 0.3], [-0.34, 0.3], [-0.34, 0.32], [-0.18, 0.32], [-0.18, 0.33], [-0.02, 0.33], [-0.02, 0.32], [0.13, 0.32], [0.13, 0.33], [0.29, 0.33], [0.29, 0.31], [0.45, 0.31], [0.45, 0.3], [0.61, 0.3], [0.61, 0.28], [0.77, 0.28], [0.77, 0.25], [0.93, 0.25], [0.93, 0.22], [1.09, 0.22], [1.09, 0.19], [1.25, 0.19], [1.25, 0.17], [1.41, 0.17], [1.41, 0.15], [1.57, 0.15], [1.57, 0.12], [1.73, 0.12], [1.73, 0.1], [1.89, 0.1], [1.89, 0.08], [2.05, 0.08], [2.05, 0.07], [2.21, 0.07], [2.21, 0.05], [2.37, 0.05], [2.37, 0.04], [2.53, 0.04], [2.53, 0.02], [2.69, 0.02], [2.69, 0.02], [2.85, 0.02], [2.85, 0.01], [3.01, 0.01], [3.01, 0.01], [3.17, 0.01], [3.17, 0.0], [3.33, 0.0], [3.33, 0.0], [3.49, 0.0], [3.49, 0.0], [3.65, 0.0], [3.65, 0.0], [3.81, 0.0], [3.81, 0.0], [3.97, 0.0], [3.97, 0.0], [4.13, 0.0], [4.13, 0.0], [4.29, 0.0], [4.29, 0.0], [4.45, 0.0], [4.45, 0.0], [4.61, 0.0], [4.61, 0.0], [4.77, 0.0], [4.77, 0.0], [4.93, 0.0], [4.93, 0.0]])
    minx = np.min(verts[:, 0])
    miny = np.min(verts[:, 1])
    maxx = np.max(verts[:, 0])
    maxy = np.max(verts[:, 1])
    p = mpath.Path(verts)
    fig, ax = plt.subplots()
    ax.add_patch(mpatches.PathPatch(p))
    ax.autoscale()
    assert ax.get_xlim()[0] <= minx
    assert ax.get_xlim()[1] >= maxx
    assert ax.get_ylim()[0] <= miny
    assert ax.get_ylim()[1] >= maxy