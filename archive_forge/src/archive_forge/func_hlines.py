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
@_preprocess_data(replace_names=['y', 'xmin', 'xmax', 'colors'], label_namer='y')
def hlines(self, y, xmin, xmax, colors=None, linestyles='solid', label='', **kwargs):
    """
        Plot horizontal lines at each *y* from *xmin* to *xmax*.

        Parameters
        ----------
        y : float or array-like
            y-indexes where to plot the lines.

        xmin, xmax : float or array-like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have the same length.

        colors : color or list of colors, default: :rc:`lines.color`

        linestyles : {'solid', 'dashed', 'dashdot', 'dotted'}, default: 'solid'

        label : str, default: ''

        Returns
        -------
        `~matplotlib.collections.LineCollection`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs :  `~matplotlib.collections.LineCollection` properties.

        See Also
        --------
        vlines : vertical lines
        axhline : horizontal line across the Axes
        """
    xmin, xmax, y = self._process_unit_info([('x', xmin), ('x', xmax), ('y', y)], kwargs)
    if not np.iterable(y):
        y = [y]
    if not np.iterable(xmin):
        xmin = [xmin]
    if not np.iterable(xmax):
        xmax = [xmax]
    y, xmin, xmax = cbook._combine_masks(y, xmin, xmax)
    y = np.ravel(y)
    xmin = np.ravel(xmin)
    xmax = np.ravel(xmax)
    masked_verts = np.ma.empty((len(y), 2, 2))
    masked_verts[:, 0, 0] = xmin
    masked_verts[:, 0, 1] = y
    masked_verts[:, 1, 0] = xmax
    masked_verts[:, 1, 1] = y
    lines = mcoll.LineCollection(masked_verts, colors=colors, linestyles=linestyles, label=label)
    self.add_collection(lines, autolim=False)
    lines._internal_update(kwargs)
    if len(y) > 0:
        updatex = True
        updatey = True
        if self.name == 'rectilinear':
            datalim = lines.get_datalim(self.transData)
            t = lines.get_transform()
            updatex, updatey = t.contains_branch_seperately(self.transData)
            minx = np.nanmin(datalim.xmin)
            maxx = np.nanmax(datalim.xmax)
            miny = np.nanmin(datalim.ymin)
            maxy = np.nanmax(datalim.ymax)
        else:
            minx = np.nanmin(masked_verts[..., 0])
            maxx = np.nanmax(masked_verts[..., 0])
            miny = np.nanmin(masked_verts[..., 1])
            maxy = np.nanmax(masked_verts[..., 1])
        corners = ((minx, miny), (maxx, maxy))
        self.update_datalim(corners, updatex, updatey)
        self._request_autoscale_view()
    return lines