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
def semilogx(self, *args, **kwargs):
    """
        Make a plot with log scaling on the x-axis.

        Call signatures::

            semilogx([x], y, [fmt], data=None, **kwargs)
            semilogx([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)

        This is just a thin wrapper around `.plot` which additionally changes
        the x-axis to log scaling. All the concepts and parameters of plot can
        be used here as well.

        The additional parameters *base*, *subs*, and *nonpositive* control the
        x-axis properties. They are just forwarded to `.Axes.set_xscale`.

        Parameters
        ----------
        base : float, default: 10
            Base of the x logarithm.

        subs : array-like, optional
            The location of the minor xticks. If *None*, reasonable locations
            are automatically chosen depending on the number of decades in the
            plot. See `.Axes.set_xscale` for details.

        nonpositive : {'mask', 'clip'}, default: 'clip'
            Non-positive values in x can be masked as invalid, or clipped to a
            very small positive number.

        **kwargs
            All parameters supported by `.plot`.

        Returns
        -------
        list of `.Line2D`
            Objects representing the plotted data.
        """
    d = {k: v for k, v in kwargs.items() if k in ['base', 'subs', 'nonpositive', 'basex', 'subsx', 'nonposx']}
    self.set_xscale('log', **d)
    return self.plot(*args, **{k: v for k, v in kwargs.items() if k not in d})