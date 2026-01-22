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
@_preprocess_data(replace_names=['x', 'weights'], label_namer='x')
def hist(self, x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, **kwargs):
    """
        Compute and plot a histogram.

        This method uses `numpy.histogram` to bin the data in *x* and count the
        number of values in each bin, then draws the distribution either as a
        `.BarContainer` or `.Polygon`. The *bins*, *range*, *density*, and
        *weights* parameters are forwarded to `numpy.histogram`.

        If the data has already been binned and counted, use `~.bar` or
        `~.stairs` to plot the distribution::

            counts, bins = np.histogram(x)
            plt.stairs(counts, bins)

        Alternatively, plot pre-computed bins and counts using ``hist()`` by
        treating each bin as a single point with a weight equal to its count::

            plt.hist(bins[:-1], bins, weights=counts)

        The data input *x* can be a singular array, a list of datasets of
        potentially different lengths ([*x0*, *x1*, ...]), or a 2D ndarray in
        which each column is a dataset. Note that the ndarray form is
        transposed relative to the list form. If the input is an array, then
        the return value is a tuple (*n*, *bins*, *patches*); if the input is a
        sequence of arrays, then the return value is a tuple
        ([*n0*, *n1*, ...], *bins*, [*patches0*, *patches1*, ...]).

        Masked arrays are not supported.

        Parameters
        ----------
        x : (n,) array or sequence of (n,) arrays
            Input values, this takes either a single array or a sequence of
            arrays which are not required to be of the same length.

        bins : int or sequence or str, default: :rc:`hist.bins`
            If *bins* is an integer, it defines the number of equal-width bins
            in the range.

            If *bins* is a sequence, it defines the bin edges, including the
            left edge of the first bin and the right edge of the last bin;
            in this case, bins may be unequally spaced.  All but the last
            (righthand-most) bin is half-open.  In other words, if *bins* is::

                [1, 2, 3, 4]

            then the first bin is ``[1, 2)`` (including 1, but excluding 2) and
            the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which
            *includes* 4.

            If *bins* is a string, it is one of the binning strategies
            supported by `numpy.histogram_bin_edges`: 'auto', 'fd', 'doane',
            'scott', 'stone', 'rice', 'sturges', or 'sqrt'.

        range : tuple or None, default: None
            The lower and upper range of the bins. Lower and upper outliers
            are ignored. If not provided, *range* is ``(x.min(), x.max())``.
            Range has no effect if *bins* is a sequence.

            If *bins* is a sequence or *range* is specified, autoscaling
            is based on the specified bin range instead of the
            range of x.

        density : bool, default: False
            If ``True``, draw and return a probability density: each bin
            will display the bin's raw count divided by the total number of
            counts *and the bin width*
            (``density = counts / (sum(counts) * np.diff(bins))``),
            so that the area under the histogram integrates to 1
            (``np.sum(density * np.diff(bins)) == 1``).

            If *stacked* is also ``True``, the sum of the histograms is
            normalized to 1.

        weights : (n,) array-like or None, default: None
            An array of weights, of the same shape as *x*.  Each value in
            *x* only contributes its associated weight towards the bin count
            (instead of 1).  If *density* is ``True``, the weights are
            normalized, so that the integral of the density over the range
            remains 1.

        cumulative : bool or -1, default: False
            If ``True``, then a histogram is computed where each bin gives the
            counts in that bin plus all bins for smaller values. The last bin
            gives the total number of datapoints.

            If *density* is also ``True`` then the histogram is normalized such
            that the last bin equals 1.

            If *cumulative* is a number less than 0 (e.g., -1), the direction
            of accumulation is reversed.  In this case, if *density* is also
            ``True``, then the histogram is normalized such that the first bin
            equals 1.

        bottom : array-like, scalar, or None, default: None
            Location of the bottom of each bin, i.e. bins are drawn from
            ``bottom`` to ``bottom + hist(x, bins)`` If a scalar, the bottom
            of each bin is shifted by the same amount. If an array, each bin
            is shifted independently and the length of bottom must match the
            number of bins. If None, defaults to 0.

        histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, default: 'bar'
            The type of histogram to draw.

            - 'bar' is a traditional bar-type histogram.  If multiple data
              are given the bars are arranged side by side.
            - 'barstacked' is a bar-type histogram where multiple
              data are stacked on top of each other.
            - 'step' generates a lineplot that is by default unfilled.
            - 'stepfilled' generates a lineplot that is by default filled.

        align : {'left', 'mid', 'right'}, default: 'mid'
            The horizontal alignment of the histogram bars.

            - 'left': bars are centered on the left bin edges.
            - 'mid': bars are centered between the bin edges.
            - 'right': bars are centered on the right bin edges.

        orientation : {'vertical', 'horizontal'}, default: 'vertical'
            If 'horizontal', `~.Axes.barh` will be used for bar-type histograms
            and the *bottom* kwarg will be the left edges.

        rwidth : float or None, default: None
            The relative width of the bars as a fraction of the bin width.  If
            ``None``, automatically compute the width.

            Ignored if *histtype* is 'step' or 'stepfilled'.

        log : bool, default: False
            If ``True``, the histogram axis will be set to a log scale.

        color : color or array-like of colors or None, default: None
            Color or sequence of colors, one per dataset.  Default (``None``)
            uses the standard line color sequence.

        label : str or None, default: None
            String, or sequence of strings to match multiple datasets.  Bar
            charts yield multiple patches per dataset, but only the first gets
            the label, so that `~.Axes.legend` will work as expected.

        stacked : bool, default: False
            If ``True``, multiple data are stacked on top of each other If
            ``False`` multiple data are arranged side by side if histtype is
            'bar' or on top of each other if histtype is 'step'

        Returns
        -------
        n : array or list of arrays
            The values of the histogram bins. See *density* and *weights* for a
            description of the possible semantics.  If input *x* is an array,
            then this is an array of length *nbins*. If input is a sequence of
            arrays ``[data1, data2, ...]``, then this is a list of arrays with
            the values of the histograms for each of the arrays in the same
            order.  The dtype of the array *n* (or of its element arrays) will
            always be float even if no weighting or normalization is used.

        bins : array
            The edges of the bins. Length nbins + 1 (nbins left edges and right
            edge of last bin).  Always a single array even when multiple data
            sets are passed in.

        patches : `.BarContainer` or list of a single `.Polygon` or list of such objects
            Container of individual artists used to create the histogram
            or list of such containers if there are multiple input datasets.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            `~matplotlib.patches.Patch` properties

        See Also
        --------
        hist2d : 2D histogram with rectangular bins
        hexbin : 2D histogram with hexagonal bins
        stairs : Plot a pre-computed histogram
        bar : Plot a pre-computed histogram

        Notes
        -----
        For large numbers of bins (>1000), plotting can be significantly
        accelerated by using `~.Axes.stairs` to plot a pre-computed histogram
        (``plt.stairs(*np.histogram(data))``), or by setting *histtype* to
        'step' or 'stepfilled' rather than 'bar' or 'barstacked'.
        """
    bin_range = range
    from builtins import range
    if np.isscalar(x):
        x = [x]
    if bins is None:
        bins = mpl.rcParams['hist.bins']
    _api.check_in_list(['bar', 'barstacked', 'step', 'stepfilled'], histtype=histtype)
    _api.check_in_list(['left', 'mid', 'right'], align=align)
    _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)
    if histtype == 'barstacked' and (not stacked):
        stacked = True
    x = cbook._reshape_2D(x, 'x')
    nx = len(x)
    if orientation == 'vertical':
        convert_units = self.convert_xunits
        x = [*self._process_unit_info([('x', x[0])], kwargs), *map(convert_units, x[1:])]
    else:
        convert_units = self.convert_yunits
        x = [*self._process_unit_info([('y', x[0])], kwargs), *map(convert_units, x[1:])]
    if bin_range is not None:
        bin_range = convert_units(bin_range)
    if not cbook.is_scalar_or_string(bins):
        bins = convert_units(bins)
    if weights is not None:
        w = cbook._reshape_2D(weights, 'weights')
    else:
        w = [None] * nx
    if len(w) != nx:
        raise ValueError('weights should have the same shape as x')
    input_empty = True
    for xi, wi in zip(x, w):
        len_xi = len(xi)
        if wi is not None and len(wi) != len_xi:
            raise ValueError('weights should have the same shape as x')
        if len_xi:
            input_empty = False
    if color is None:
        colors = [self._get_lines.get_next_color() for i in range(nx)]
    else:
        colors = mcolors.to_rgba_array(color)
        if len(colors) != nx:
            raise ValueError(f"The 'color' keyword argument must have one color per dataset, but {nx} datasets and {len(colors)} colors were provided")
    hist_kwargs = dict()
    if bin_range is None:
        xmin = np.inf
        xmax = -np.inf
        for xi in x:
            if len(xi):
                xmin = min(xmin, np.nanmin(xi))
                xmax = max(xmax, np.nanmax(xi))
        if xmin <= xmax:
            bin_range = (xmin, xmax)
    if not input_empty and len(x) > 1:
        if weights is not None:
            _w = np.concatenate(w)
        else:
            _w = None
        bins = np.histogram_bin_edges(np.concatenate(x), bins, bin_range, _w)
    else:
        hist_kwargs['range'] = bin_range
    density = bool(density)
    if density and (not stacked):
        hist_kwargs['density'] = density
    tops = []
    for i in range(nx):
        m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
        tops.append(m)
    tops = np.array(tops, float)
    bins = np.array(bins, float)
    if stacked:
        tops = tops.cumsum(axis=0)
        if density:
            tops = tops / np.diff(bins) / tops[-1].sum()
    if cumulative:
        slc = slice(None)
        if isinstance(cumulative, Number) and cumulative < 0:
            slc = slice(None, None, -1)
        if density:
            tops = (tops * np.diff(bins))[:, slc].cumsum(axis=1)[:, slc]
        else:
            tops = tops[:, slc].cumsum(axis=1)[:, slc]
    patches = []
    if histtype.startswith('bar'):
        totwidth = np.diff(bins)
        if rwidth is not None:
            dr = np.clip(rwidth, 0, 1)
        elif len(tops) > 1 and (not stacked or mpl.rcParams['_internal.classic_mode']):
            dr = 0.8
        else:
            dr = 1.0
        if histtype == 'bar' and (not stacked):
            width = dr * totwidth / nx
            dw = width
            boffset = -0.5 * dr * totwidth * (1 - 1 / nx)
        elif histtype == 'barstacked' or stacked:
            width = dr * totwidth
            boffset, dw = (0.0, 0.0)
        if align == 'mid':
            boffset += 0.5 * totwidth
        elif align == 'right':
            boffset += totwidth
        if orientation == 'horizontal':
            _barfunc = self.barh
            bottom_kwarg = 'left'
        else:
            _barfunc = self.bar
            bottom_kwarg = 'bottom'
        for top, color in zip(tops, colors):
            if bottom is None:
                bottom = np.zeros(len(top))
            if stacked:
                height = top - bottom
            else:
                height = top
            bars = _barfunc(bins[:-1] + boffset, height, width, align='center', log=log, color=color, **{bottom_kwarg: bottom})
            patches.append(bars)
            if stacked:
                bottom = top
            boffset += dw
        for bars in patches[1:]:
            for patch in bars:
                patch.sticky_edges.x[:] = patch.sticky_edges.y[:] = []
    elif histtype.startswith('step'):
        x = np.zeros(4 * len(bins) - 3)
        y = np.zeros(4 * len(bins) - 3)
        x[0:2 * len(bins) - 1:2], x[1:2 * len(bins) - 1:2] = (bins, bins[:-1])
        x[2 * len(bins) - 1:] = x[1:2 * len(bins) - 1][::-1]
        if bottom is None:
            bottom = 0
        y[1:2 * len(bins) - 1:2] = y[2:2 * len(bins):2] = bottom
        y[2 * len(bins) - 1:] = y[1:2 * len(bins) - 1][::-1]
        if log:
            if orientation == 'horizontal':
                self.set_xscale('log', nonpositive='clip')
            else:
                self.set_yscale('log', nonpositive='clip')
        if align == 'left':
            x -= 0.5 * (bins[1] - bins[0])
        elif align == 'right':
            x += 0.5 * (bins[1] - bins[0])
        fill = histtype == 'stepfilled'
        xvals, yvals = ([], [])
        for top in tops:
            if stacked:
                y[2 * len(bins) - 1:] = y[1:2 * len(bins) - 1][::-1]
            y[1:2 * len(bins) - 1:2] = y[2:2 * len(bins):2] = top + bottom
            y[0] = y[-1]
            if orientation == 'horizontal':
                xvals.append(y.copy())
                yvals.append(x.copy())
            else:
                xvals.append(x.copy())
                yvals.append(y.copy())
        split = -1 if fill else 2 * len(bins)
        for x, y, color in reversed(list(zip(xvals, yvals, colors))):
            patches.append(self.fill(x[:split], y[:split], closed=True if fill else None, facecolor=color, edgecolor=None if fill else color, fill=fill if fill else None, zorder=None if fill else mlines.Line2D.zorder))
        for patch_list in patches:
            for patch in patch_list:
                if orientation == 'vertical':
                    patch.sticky_edges.y.append(0)
                elif orientation == 'horizontal':
                    patch.sticky_edges.x.append(0)
        patches.reverse()
    labels = [] if label is None else np.atleast_1d(np.asarray(label, str))
    for patch, lbl in itertools.zip_longest(patches, labels):
        if patch:
            p = patch[0]
            p._internal_update(kwargs)
            if lbl is not None:
                p.set_label(lbl)
            for p in patch[1:]:
                p._internal_update(kwargs)
                p.set_label('_nolegend_')
    if nx == 1:
        return (tops[0], bins, patches[0])
    else:
        patch_type = 'BarContainer' if histtype.startswith('bar') else 'list[Polygon]'
        return (tops, bins, cbook.silent_list(patch_type, patches))