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
def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs):
    """
        Add a horizontal span (rectangle) across the Axes.

        The rectangle spans from *ymin* to *ymax* vertically, and, by default,
        the whole x-axis horizontally.  The x-span can be set using *xmin*
        (default: 0) and *xmax* (default: 1) which are in axis units; e.g.
        ``xmin = 0.5`` always refers to the middle of the x-axis regardless of
        the limits set by `~.Axes.set_xlim`.

        Parameters
        ----------
        ymin : float
            Lower y-coordinate of the span, in data units.
        ymax : float
            Upper y-coordinate of the span, in data units.
        xmin : float, default: 0
            Lower x-coordinate of the span, in x-axis (0-1) units.
        xmax : float, default: 1
            Upper x-coordinate of the span, in x-axis (0-1) units.

        Returns
        -------
        `~matplotlib.patches.Polygon`
            Horizontal span (rectangle) from (xmin, ymin) to (xmax, ymax).

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Polygon` properties

        %(Polygon:kwdoc)s

        See Also
        --------
        axvspan : Add a vertical span across the Axes.
        """
    self._check_no_units([xmin, xmax], ['xmin', 'xmax'])
    (ymin, ymax), = self._process_unit_info([('y', [ymin, ymax])], kwargs)
    verts = ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin))
    p = mpatches.Polygon(verts, **kwargs)
    p.set_transform(self.get_yaxis_transform(which='grid'))
    self.add_patch(p)
    self._request_autoscale_view('y')
    return p