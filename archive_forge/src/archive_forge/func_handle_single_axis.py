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
def handle_single_axis(scale, shared_axes, name, axis, margin, stickies, set_bound):
    if not (scale and axis._get_autoscale_on()):
        return
    shared = shared_axes.get_siblings(self)
    values = [val for ax in shared for val in getattr(ax.dataLim, f'interval{name}') if np.isfinite(val)]
    if values:
        x0, x1 = (min(values), max(values))
    elif getattr(self._viewLim, f'mutated{name}')():
        return
    else:
        x0, x1 = (-np.inf, np.inf)
    locator = axis.get_major_locator()
    x0, x1 = locator.nonsingular(x0, x1)
    minimum_minpos = min((getattr(ax.dataLim, f'minpos{name}') for ax in shared))
    tol = 1e-05 * max(abs(x0), abs(x1), abs(x1 - x0))
    i0 = stickies.searchsorted(x0 + tol) - 1
    x0bound = stickies[i0] if i0 != -1 else None
    i1 = stickies.searchsorted(x1 - tol)
    x1bound = stickies[i1] if i1 != len(stickies) else None
    transform = axis.get_transform()
    inverse_trans = transform.inverted()
    x0, x1 = axis._scale.limit_range_for_scale(x0, x1, minimum_minpos)
    x0t, x1t = transform.transform([x0, x1])
    delta = (x1t - x0t) * margin
    if not np.isfinite(delta):
        delta = 0
    x0, x1 = inverse_trans.transform([x0t - delta, x1t + delta])
    if x0bound is not None:
        x0 = max(x0, x0bound)
    if x1bound is not None:
        x1 = min(x1, x1bound)
    if not self._tight:
        x0, x1 = locator.view_limits(x0, x1)
    set_bound(x0, x1)