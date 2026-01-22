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
def test_shared_with_aspect_3():
    for adjustable in ['box', 'datalim']:
        fig, axs = plt.subplots(nrows=2, sharey=True)
        axs[0].set_aspect(2, adjustable=adjustable)
        axs[1].set_aspect(0.5, adjustable=adjustable)
        axs[0].plot([1, 2], [3, 4])
        axs[1].plot([3, 4], [1, 2])
        plt.draw()
        assert axs[0].get_xlim() != axs[1].get_xlim()
        assert axs[0].get_ylim() == axs[1].get_ylim()
        fig_aspect = fig.bbox_inches.height / fig.bbox_inches.width
        for ax in axs:
            p = ax.get_position()
            box_aspect = p.height / p.width
            lim_aspect = ax.viewLim.height / ax.viewLim.width
            expected = fig_aspect * box_aspect / lim_aspect
            assert round(expected, 4) == round(ax.get_aspect(), 4)