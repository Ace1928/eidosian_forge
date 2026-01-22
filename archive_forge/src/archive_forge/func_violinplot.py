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
@_preprocess_data(replace_names=['dataset'])
def violinplot(self, dataset, positions=None, vert=True, widths=0.5, showmeans=False, showextrema=True, showmedians=False, quantiles=None, points=100, bw_method=None):
    """
        Make a violin plot.

        Make a violin plot for each column of *dataset* or each vector in
        sequence *dataset*.  Each filled area extends to represent the
        entire data range, with optional lines at the mean, the median,
        the minimum, the maximum, and user-specified quantiles.

        Parameters
        ----------
        dataset : Array or a sequence of vectors.
          The input data.

        positions : array-like, default: [1, 2, ..., n]
          The positions of the violins. The ticks and limits are
          automatically set to match the positions.

        vert : bool, default: True.
          If true, creates a vertical violin plot.
          Otherwise, creates a horizontal violin plot.

        widths : array-like, default: 0.5
          Either a scalar or a vector that sets the maximal width of
          each violin. The default is 0.5, which uses about half of the
          available horizontal space.

        showmeans : bool, default: False
          If `True`, will toggle rendering of the means.

        showextrema : bool, default: True
          If `True`, will toggle rendering of the extrema.

        showmedians : bool, default: False
          If `True`, will toggle rendering of the medians.

        quantiles : array-like, default: None
          If not None, set a list of floats in interval [0, 1] for each violin,
          which stands for the quantiles that will be rendered for that
          violin.

        points : int, default: 100
          Defines the number of points to evaluate each of the
          gaussian kernel density estimations at.

        bw_method : str, scalar or callable, optional
          The method used to calculate the estimator bandwidth.  This can be
          'scott', 'silverman', a scalar constant or a callable.  If a
          scalar, this will be used directly as `kde.factor`.  If a
          callable, it should take a `matplotlib.mlab.GaussianKDE` instance as
          its only parameter and return a scalar. If None (default), 'scott'
          is used.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        Returns
        -------
        dict
          A dictionary mapping each component of the violinplot to a
          list of the corresponding collection instances created. The
          dictionary has the following keys:

          - ``bodies``: A list of the `~.collections.PolyCollection`
            instances containing the filled area of each violin.

          - ``cmeans``: A `~.collections.LineCollection` instance that marks
            the mean values of each of the violin's distribution.

          - ``cmins``: A `~.collections.LineCollection` instance that marks
            the bottom of each violin's distribution.

          - ``cmaxes``: A `~.collections.LineCollection` instance that marks
            the top of each violin's distribution.

          - ``cbars``: A `~.collections.LineCollection` instance that marks
            the centers of each violin's distribution.

          - ``cmedians``: A `~.collections.LineCollection` instance that
            marks the median values of each of the violin's distribution.

          - ``cquantiles``: A `~.collections.LineCollection` instance created
            to identify the quantile values of each of the violin's
            distribution.

        """

    def _kde_method(X, coords):
        X = cbook._unpack_to_numpy(X)
        if np.all(X[0] == X):
            return (X[0] == coords).astype(float)
        kde = mlab.GaussianKDE(X, bw_method)
        return kde.evaluate(coords)
    vpstats = cbook.violin_stats(dataset, _kde_method, points=points, quantiles=quantiles)
    return self.violin(vpstats, positions=positions, vert=vert, widths=widths, showmeans=showmeans, showextrema=showextrema, showmedians=showmedians)