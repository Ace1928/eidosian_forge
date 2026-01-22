from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def subfigures(self, nrows=1, ncols=1, squeeze=True, wspace=None, hspace=None, width_ratios=None, height_ratios=None, **kwargs):
    """
        Add a set of subfigures to this figure or subfigure.

        A subfigure has the same artist methods as a figure, and is logically
        the same as a figure, but cannot print itself.
        See :doc:`/gallery/subplots_axes_and_figures/subfigures`.

        .. note::
            The *subfigure* concept is new in v3.4, and the API is still provisional.

        Parameters
        ----------
        nrows, ncols : int, default: 1
            Number of rows/columns of the subfigure grid.

        squeeze : bool, default: True
            If True, extra dimensions are squeezed out from the returned
            array of subfigures.

        wspace, hspace : float, default: None
            The amount of width/height reserved for space between subfigures,
            expressed as a fraction of the average subfigure width/height.
            If not given, the values will be inferred from rcParams if using
            constrained layout (see `~.ConstrainedLayoutEngine`), or zero if
            not using a layout engine.

        width_ratios : array-like of length *ncols*, optional
            Defines the relative widths of the columns. Each column gets a
            relative width of ``width_ratios[i] / sum(width_ratios)``.
            If not given, all columns will have the same width.

        height_ratios : array-like of length *nrows*, optional
            Defines the relative heights of the rows. Each row gets a
            relative height of ``height_ratios[i] / sum(height_ratios)``.
            If not given, all rows will have the same height.
        """
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=self, wspace=wspace, hspace=hspace, width_ratios=width_ratios, height_ratios=height_ratios, left=0, right=1, bottom=0, top=1)
    sfarr = np.empty((nrows, ncols), dtype=object)
    for i in range(ncols):
        for j in range(nrows):
            sfarr[j, i] = self.add_subfigure(gs[j, i], **kwargs)
    if self.get_layout_engine() is None and (wspace is not None or hspace is not None):
        bottoms, tops, lefts, rights = gs.get_grid_positions(self)
        for sfrow, bottom, top in zip(sfarr, bottoms, tops):
            for sf, left, right in zip(sfrow, lefts, rights):
                bbox = Bbox.from_extents(left, bottom, right, top)
                sf._redo_transform_rel_fig(bbox=bbox)
    if squeeze:
        return sfarr.item() if sfarr.size == 1 else sfarr.squeeze()
    else:
        return sfarr