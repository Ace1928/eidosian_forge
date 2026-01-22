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
@_docstring.dedent_interpd
def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
    """
        Add a vertical line across the Axes.

        Parameters
        ----------
        x : float, default: 0
            x position in data coordinates of the vertical line.

        ymin : float, default: 0
            Should be between 0 and 1, 0 being the bottom of the plot, 1 the
            top of the plot.

        ymax : float, default: 1
            Should be between 0 and 1, 0 being the bottom of the plot, 1 the
            top of the plot.

        Returns
        -------
        `~matplotlib.lines.Line2D`

        Other Parameters
        ----------------
        **kwargs
            Valid keyword arguments are `.Line2D` properties, except for
            'transform':

            %(Line2D:kwdoc)s

        See Also
        --------
        vlines : Add vertical lines in data coordinates.
        axvspan : Add a vertical span (rectangle) across the axis.
        axline : Add a line with an arbitrary slope.

        Examples
        --------
        * draw a thick red vline at *x* = 0 that spans the yrange::

            >>> axvline(linewidth=4, color='r')

        * draw a default vline at *x* = 1 that spans the yrange::

            >>> axvline(x=1)

        * draw a default vline at *x* = .5 that spans the middle half of
          the yrange::

            >>> axvline(x=.5, ymin=0.25, ymax=0.75)
        """
    self._check_no_units([ymin, ymax], ['ymin', 'ymax'])
    if 'transform' in kwargs:
        raise ValueError("'transform' is not allowed as a keyword argument; axvline generates its own transform.")
    xmin, xmax = self.get_xbound()
    xx, = self._process_unit_info([('x', x)], kwargs)
    scalex = xx < xmin or xx > xmax
    trans = self.get_xaxis_transform(which='grid')
    l = mlines.Line2D([x, x], [ymin, ymax], transform=trans, **kwargs)
    self.add_line(l)
    if scalex:
        self._request_autoscale_view('x')
    return l