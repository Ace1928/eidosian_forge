from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def sharey(self, other):
    """
        Share the y-axis with *other*.

        This is equivalent to passing ``sharey=other`` when constructing the
        Axes, and cannot be used if the y-axis is already being shared with
        another Axes.
        """
    _api.check_isinstance(_AxesBase, other=other)
    if self._sharey is not None and other is not self._sharey:
        raise ValueError('y-axis is already shared')
    self._shared_axes['y'].join(self, other)
    self._sharey = other
    self.yaxis.major = other.yaxis.major
    self.yaxis.minor = other.yaxis.minor
    y0, y1 = other.get_ylim()
    self.set_ylim(y0, y1, emit=False, auto=other.get_autoscaley_on())
    self.yaxis._scale = other.yaxis._scale