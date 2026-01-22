import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
@_preprocess_data(replace_names=['positions', 'lineoffsets', 'linelengths', 'linewidths', 'colors', 'linestyles'])
@_docstring.dedent_interpd
def eventplot(self, positions, orientation='horizontal', lineoffsets=1, linelengths=1, linewidths=None, colors=None, alpha=None, linestyles='solid', **kwargs):
    """
        Plot identical parallel lines at the given positions.

        This type of plot is commonly used in neuroscience for representing
        neural events, where it is usually called a spike raster, dot raster,
        or raster plot.

        However, it is useful in any situation where you wish to show the
        timing or position of multiple sets of discrete events, such as the
        arrival times of people to a business on each day of the month or the
        date of hurricanes each year of the last century.

        Parameters
        ----------
        positions : array-like or list of array-like
            A 1D array-like defines the positions of one sequence of events.

            Multiple groups of events may be passed as a list of array-likes.
            Each group can be styled independently by passing lists of values
            to *lineoffsets*, *linelengths*, *linewidths*, *colors* and
            *linestyles*.

            Note that *positions* can be a 2D array, but in practice different
            event groups usually have different counts so that one will use a
            list of different-length arrays rather than a 2D array.

        orientation : {'horizontal', 'vertical'}, default: 'horizontal'
            The direction of the event sequence:

            - 'horizontal': the events are arranged horizontally.
              The indicator lines are vertical.
            - 'vertical': the events are arranged vertically.
              The indicator lines are horizontal.

        lineoffsets : float or array-like, default: 1
            The offset of the center of the lines from the origin, in the
            direction orthogonal to *orientation*.

            If *positions* is 2D, this can be a sequence with length matching
            the length of *positions*.

        linelengths : float or array-like, default: 1
            The total height of the lines (i.e. the lines stretches from
            ``lineoffset - linelength/2`` to ``lineoffset + linelength/2``).

            If *positions* is 2D, this can be a sequence with length matching
            the length of *positions*.

        linewidths : float or array-like, default: :rc:`lines.linewidth`
            The line width(s) of the event lines, in points.

            If *positions* is 2D, this can be a sequence with length matching
            the length of *positions*.

        colors : color or list of colors, default: :rc:`lines.color`
            The color(s) of the event lines.

            If *positions* is 2D, this can be a sequence with length matching
            the length of *positions*.

        alpha : float or array-like, default: 1
            The alpha blending value(s), between 0 (transparent) and 1
            (opaque).

            If *positions* is 2D, this can be a sequence with length matching
            the length of *positions*.

        linestyles : str or tuple or list of such values, default: 'solid'
            Default is 'solid'. Valid strings are ['solid', 'dashed',
            'dashdot', 'dotted', '-', '--', '-.', ':']. Dash tuples
            should be of the form::

                (offset, onoffseq),

            where *onoffseq* is an even length tuple of on and off ink
            in points.

            If *positions* is 2D, this can be a sequence with length matching
            the length of *positions*.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Other keyword arguments are line collection properties.  See
            `.LineCollection` for a list of the valid properties.

        Returns
        -------
        list of `.EventCollection`
            The `.EventCollection` that were added.

        Notes
        -----
        For *linelengths*, *linewidths*, *colors*, *alpha* and *linestyles*, if
        only a single value is given, that value is applied to all lines. If an
        array-like is given, it must have the same length as *positions*, and
        each value will be applied to the corresponding row of the array.

        Examples
        --------
        .. plot:: gallery/lines_bars_and_markers/eventplot_demo.py
        """
    lineoffsets, linelengths = self._process_unit_info([('y', lineoffsets), ('y', linelengths)], kwargs)
    if not np.iterable(positions):
        positions = [positions]
    elif any((np.iterable(position) for position in positions)):
        positions = [np.asanyarray(position) for position in positions]
    else:
        positions = [np.asanyarray(positions)]
    poss = []
    for position in positions:
        poss += self._process_unit_info([('x', position)], kwargs)
    positions = poss
    colors = cbook._local_over_kwdict(colors, kwargs, 'color')
    linewidths = cbook._local_over_kwdict(linewidths, kwargs, 'linewidth')
    linestyles = cbook._local_over_kwdict(linestyles, kwargs, 'linestyle')
    if not np.iterable(lineoffsets):
        lineoffsets = [lineoffsets]
    if not np.iterable(linelengths):
        linelengths = [linelengths]
    if not np.iterable(linewidths):
        linewidths = [linewidths]
    if not np.iterable(colors):
        colors = [colors]
    if not np.iterable(alpha):
        alpha = [alpha]
    if hasattr(linestyles, 'lower') or not np.iterable(linestyles):
        linestyles = [linestyles]
    lineoffsets = np.asarray(lineoffsets)
    linelengths = np.asarray(linelengths)
    linewidths = np.asarray(linewidths)
    if len(lineoffsets) == 0:
        raise ValueError('lineoffsets cannot be empty')
    if len(linelengths) == 0:
        raise ValueError('linelengths cannot be empty')
    if len(linestyles) == 0:
        raise ValueError('linestyles cannot be empty')
    if len(linewidths) == 0:
        raise ValueError('linewidths cannot be empty')
    if len(alpha) == 0:
        raise ValueError('alpha cannot be empty')
    if len(colors) == 0:
        colors = [None]
    try:
        colors = mcolors.to_rgba_array(colors)
    except ValueError:
        pass
    if len(lineoffsets) == 1 and len(positions) != 1:
        lineoffsets = np.tile(lineoffsets, len(positions))
        lineoffsets[0] = 0
        lineoffsets = np.cumsum(lineoffsets)
    if len(linelengths) == 1:
        linelengths = np.tile(linelengths, len(positions))
    if len(linewidths) == 1:
        linewidths = np.tile(linewidths, len(positions))
    if len(colors) == 1:
        colors = list(colors) * len(positions)
    if len(alpha) == 1:
        alpha = list(alpha) * len(positions)
    if len(linestyles) == 1:
        linestyles = [linestyles] * len(positions)
    if len(lineoffsets) != len(positions):
        raise ValueError('lineoffsets and positions are unequal sized sequences')
    if len(linelengths) != len(positions):
        raise ValueError('linelengths and positions are unequal sized sequences')
    if len(linewidths) != len(positions):
        raise ValueError('linewidths and positions are unequal sized sequences')
    if len(colors) != len(positions):
        raise ValueError('colors and positions are unequal sized sequences')
    if len(alpha) != len(positions):
        raise ValueError('alpha and positions are unequal sized sequences')
    if len(linestyles) != len(positions):
        raise ValueError('linestyles and positions are unequal sized sequences')
    colls = []
    for position, lineoffset, linelength, linewidth, color, alpha_, linestyle in zip(positions, lineoffsets, linelengths, linewidths, colors, alpha, linestyles):
        coll = mcoll.EventCollection(position, orientation=orientation, lineoffset=lineoffset, linelength=linelength, linewidth=linewidth, color=color, alpha=alpha_, linestyle=linestyle)
        self.add_collection(coll, autolim=False)
        coll._internal_update(kwargs)
        colls.append(coll)
    if len(positions) > 0:
        min_max = [(np.min(_p), np.max(_p)) for _p in positions if len(_p) > 0]
        if len(min_max) > 0:
            mins, maxes = zip(*min_max)
            minpos = np.min(mins)
            maxpos = np.max(maxes)
            minline = (lineoffsets - linelengths).min()
            maxline = (lineoffsets + linelengths).max()
            if orientation == 'vertical':
                corners = ((minline, minpos), (maxline, maxpos))
            else:
                corners = ((minpos, minline), (maxpos, maxline))
            self.update_datalim(corners)
            self._request_autoscale_view()
    return colls