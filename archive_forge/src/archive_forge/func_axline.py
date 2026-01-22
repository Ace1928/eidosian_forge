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
def axline(self, xy1, xy2=None, *, slope=None, **kwargs):
    """
        Add an infinitely long straight line.

        The line can be defined either by two points *xy1* and *xy2*, or
        by one point *xy1* and a *slope*.

        This draws a straight line "on the screen", regardless of the x and y
        scales, and is thus also suitable for drawing exponential decays in
        semilog plots, power laws in loglog plots, etc. However, *slope*
        should only be used with linear scales; It has no clear meaning for
        all other scales, and thus the behavior is undefined. Please specify
        the line using the points *xy1*, *xy2* for non-linear scales.

        The *transform* keyword argument only applies to the points *xy1*,
        *xy2*. The *slope* (if given) is always in data coordinates. This can
        be used e.g. with ``ax.transAxes`` for drawing grid lines with a fixed
        slope.

        Parameters
        ----------
        xy1, xy2 : (float, float)
            Points for the line to pass through.
            Either *xy2* or *slope* has to be given.
        slope : float, optional
            The slope of the line. Either *xy2* or *slope* has to be given.

        Returns
        -------
        `.AxLine`

        Other Parameters
        ----------------
        **kwargs
            Valid kwargs are `.Line2D` properties

            %(Line2D:kwdoc)s

        See Also
        --------
        axhline : for horizontal lines
        axvline : for vertical lines

        Examples
        --------
        Draw a thick red line passing through (0, 0) and (1, 1)::

            >>> axline((0, 0), (1, 1), linewidth=4, color='r')
        """
    if slope is not None and (self.get_xscale() != 'linear' or self.get_yscale() != 'linear'):
        raise TypeError("'slope' cannot be used with non-linear scales")
    datalim = [xy1] if xy2 is None else [xy1, xy2]
    if 'transform' in kwargs:
        datalim = []
    line = mlines.AxLine(xy1, xy2, slope, **kwargs)
    self._set_artist_props(line)
    if line.get_clip_path() is None:
        line.set_clip_path(self.patch)
    if not line.get_label():
        line.set_label(f'_child{len(self._children)}')
    self._children.append(line)
    line._remove_method = self._children.remove
    self.update_datalim(datalim)
    self._request_autoscale_view()
    return line