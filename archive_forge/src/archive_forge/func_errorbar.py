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
@_preprocess_data(replace_names=['x', 'y', 'xerr', 'yerr'], label_namer='y')
@_docstring.dedent_interpd
def errorbar(self, x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, **kwargs):
    """
        Plot y versus x as lines and/or markers with attached errorbars.

        *x*, *y* define the data locations, *xerr*, *yerr* define the errorbar
        sizes. By default, this draws the data markers/lines as well as the
        errorbars. Use fmt='none' to draw errorbars without any data markers.

        .. versionadded:: 3.7
           Caps and error lines are drawn in polar coordinates on polar plots.


        Parameters
        ----------
        x, y : float or array-like
            The data positions.

        xerr, yerr : float or array-like, shape(N,) or shape(2, N), optional
            The errorbar sizes:

            - scalar: Symmetric +/- values for all data points.
            - shape(N,): Symmetric +/-values for each data point.
            - shape(2, N): Separate - and + values for each bar. First row
              contains the lower errors, the second row contains the upper
              errors.
            - *None*: No errorbar.

            All values must be >= 0.

            See :doc:`/gallery/statistics/errorbar_features`
            for an example on the usage of ``xerr`` and ``yerr``.

        fmt : str, default: ''
            The format for the data points / data lines. See `.plot` for
            details.

            Use 'none' (case-insensitive) to plot errorbars without any data
            markers.

        ecolor : color, default: None
            The color of the errorbar lines.  If None, use the color of the
            line connecting the markers.

        elinewidth : float, default: None
            The linewidth of the errorbar lines. If None, the linewidth of
            the current style is used.

        capsize : float, default: :rc:`errorbar.capsize`
            The length of the error bar caps in points.

        capthick : float, default: None
            An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).
            This setting is a more sensible name for the property that
            controls the thickness of the error bar cap in points. For
            backwards compatibility, if *mew* or *markeredgewidth* are given,
            then they will over-ride *capthick*. This may change in future
            releases.

        barsabove : bool, default: False
            If True, will plot the errorbars above the plot
            symbols. Default is below.

        lolims, uplims, xlolims, xuplims : bool or array-like, default: False
            These arguments can be used to indicate that a value gives only
            upper/lower limits.  In that case a caret symbol is used to
            indicate this. *lims*-arguments may be scalars, or array-likes of
            the same length as *xerr* and *yerr*.  To use limits with inverted
            axes, `~.Axes.set_xlim` or `~.Axes.set_ylim` must be called before
            :meth:`errorbar`.  Note the tricky parameter names: setting e.g.
            *lolims* to True means that the y-value is a *lower* limit of the
            True value, so, only an *upward*-pointing arrow will be drawn!

        errorevery : int or (int, int), default: 1
            draws error bars on a subset of the data. *errorevery* =N draws
            error bars on the points (x[::N], y[::N]).
            *errorevery* =(start, N) draws error bars on the points
            (x[start::N], y[start::N]). e.g. errorevery=(6, 3)
            adds error bars to the data at (x[6], x[9], x[12], x[15], ...).
            Used to avoid overlapping error bars when two series share x-axis
            values.

        Returns
        -------
        `.ErrorbarContainer`
            The container contains:

            - plotline: `~matplotlib.lines.Line2D` instance of x, y plot markers
              and/or line.
            - caplines: A tuple of `~matplotlib.lines.Line2D` instances of the error
              bar caps.
            - barlinecols: A tuple of `.LineCollection` with the horizontal and
              vertical error ranges.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            All other keyword arguments are passed on to the `~.Axes.plot` call
            drawing the markers. For example, this code makes big red squares
            with thick green edges::

                x, y, yerr = rand(3, 10)
                errorbar(x, y, yerr, marker='s', mfc='red',
                         mec='green', ms=20, mew=4)

            where *mfc*, *mec*, *ms* and *mew* are aliases for the longer
            property names, *markerfacecolor*, *markeredgecolor*, *markersize*
            and *markeredgewidth*.

            Valid kwargs for the marker properties are:

            - *dashes*
            - *dash_capstyle*
            - *dash_joinstyle*
            - *drawstyle*
            - *fillstyle*
            - *linestyle*
            - *marker*
            - *markeredgecolor*
            - *markeredgewidth*
            - *markerfacecolor*
            - *markerfacecoloralt*
            - *markersize*
            - *markevery*
            - *solid_capstyle*
            - *solid_joinstyle*

            Refer to the corresponding `.Line2D` property for more details:

            %(Line2D:kwdoc)s
        """
    kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs.setdefault('zorder', 2)
    if not isinstance(x, np.ndarray):
        x = np.asarray(x, dtype=object)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y, dtype=object)

    def _upcast_err(err):
        """
            Safely handle tuple of containers that carry units.

            This function covers the case where the input to the xerr/yerr is a
            length 2 tuple of equal length ndarray-subclasses that carry the
            unit information in the container.

            If we have a tuple of nested numpy array (subclasses), we defer
            coercing the units to be consistent to the underlying unit
            library (and implicitly the broadcasting).

            Otherwise, fallback to casting to an object array.
            """
        if np.iterable(err) and len(err) > 0 and isinstance(cbook._safe_first_finite(err), np.ndarray):
            atype = type(cbook._safe_first_finite(err))
            if atype is np.ndarray:
                return np.asarray(err, dtype=object)
            return atype(err)
        return np.asarray(err, dtype=object)
    if xerr is not None and (not isinstance(xerr, np.ndarray)):
        xerr = _upcast_err(xerr)
    if yerr is not None and (not isinstance(yerr, np.ndarray)):
        yerr = _upcast_err(yerr)
    x, y = np.atleast_1d(x, y)
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same size")
    everymask = self._errorevery_to_mask(x, errorevery)
    label = kwargs.pop('label', None)
    kwargs['label'] = '_nolegend_'
    (data_line, base_style), = self._get_lines._plot_args(self, (x, y) if fmt == '' else (x, y, fmt), kwargs, return_kwargs=True)
    if barsabove:
        data_line.set_zorder(kwargs['zorder'] - 0.1)
    else:
        data_line.set_zorder(kwargs['zorder'] + 0.1)
    if fmt.lower() != 'none':
        self.add_line(data_line)
    else:
        data_line = None
        base_style.pop('color')
        if 'color' in kwargs:
            base_style['color'] = kwargs.pop('color')
    if 'color' not in base_style:
        base_style['color'] = 'C0'
    if ecolor is None:
        ecolor = base_style['color']
    for key in ['marker', 'markersize', 'markerfacecolor', 'markerfacecoloralt', 'markeredgewidth', 'markeredgecolor', 'markevery', 'linestyle', 'fillstyle', 'drawstyle', 'dash_capstyle', 'dash_joinstyle', 'solid_capstyle', 'solid_joinstyle', 'dashes']:
        base_style.pop(key, None)
    eb_lines_style = {**base_style, 'color': ecolor}
    if elinewidth is not None:
        eb_lines_style['linewidth'] = elinewidth
    elif 'linewidth' in kwargs:
        eb_lines_style['linewidth'] = kwargs['linewidth']
    for key in ('transform', 'alpha', 'zorder', 'rasterized'):
        if key in kwargs:
            eb_lines_style[key] = kwargs[key]
    eb_cap_style = {**base_style, 'linestyle': 'none'}
    if capsize is None:
        capsize = mpl.rcParams['errorbar.capsize']
    if capsize > 0:
        eb_cap_style['markersize'] = 2.0 * capsize
    if capthick is not None:
        eb_cap_style['markeredgewidth'] = capthick
    for key in ('markeredgewidth', 'transform', 'alpha', 'zorder', 'rasterized'):
        if key in kwargs:
            eb_cap_style[key] = kwargs[key]
    eb_cap_style['color'] = ecolor
    barcols = []
    caplines = {'x': [], 'y': []}

    def apply_mask(arrays, mask):
        return [array[mask] for array in arrays]
    for dep_axis, dep, err, lolims, uplims, indep, lines_func, marker, lomarker, himarker in [('x', x, xerr, xlolims, xuplims, y, self.hlines, '|', mlines.CARETRIGHTBASE, mlines.CARETLEFTBASE), ('y', y, yerr, lolims, uplims, x, self.vlines, '_', mlines.CARETUPBASE, mlines.CARETDOWNBASE)]:
        if err is None:
            continue
        lolims = np.broadcast_to(lolims, len(dep)).astype(bool)
        uplims = np.broadcast_to(uplims, len(dep)).astype(bool)
        try:
            np.broadcast_to(err, (2, len(dep)))
        except ValueError:
            raise ValueError(f"'{dep_axis}err' (shape: {np.shape(err)}) must be a scalar or a 1D or (2, n) array-like whose shape matches '{dep_axis}' (shape: {np.shape(dep)})") from None
        res = np.zeros(err.shape, dtype=bool)
        if np.any(np.less(err, -err, out=res, where=err == err)):
            raise ValueError(f"'{dep_axis}err' must not contain negative values")
        low, high = dep + np.vstack([-(1 - lolims), 1 - uplims]) * err
        barcols.append(lines_func(*apply_mask([indep, low, high], everymask), **eb_lines_style))
        if self.name == 'polar' and dep_axis == 'x':
            for b in barcols:
                for p in b.get_paths():
                    p._interpolation_steps = 2
        nolims = ~(lolims | uplims)
        if nolims.any() and capsize > 0:
            indep_masked, lo_masked, hi_masked = apply_mask([indep, low, high], nolims & everymask)
            for lh_masked in [lo_masked, hi_masked]:
                line = mlines.Line2D(indep_masked, indep_masked, marker=marker, **eb_cap_style)
                line.set(**{f'{dep_axis}data': lh_masked})
                caplines[dep_axis].append(line)
        for idx, (lims, hl) in enumerate([(lolims, high), (uplims, low)]):
            if not lims.any():
                continue
            hlmarker = himarker if self._axis_map[dep_axis].get_inverted() ^ idx else lomarker
            x_masked, y_masked, hl_masked = apply_mask([x, y, hl], lims & everymask)
            line = mlines.Line2D(x_masked, y_masked, marker=hlmarker, **eb_cap_style)
            line.set(**{f'{dep_axis}data': hl_masked})
            caplines[dep_axis].append(line)
            if capsize > 0:
                caplines[dep_axis].append(mlines.Line2D(x_masked, y_masked, marker=marker, **eb_cap_style))
    if self.name == 'polar':
        for axis in caplines:
            for l in caplines[axis]:
                for theta, r in zip(l.get_xdata(), l.get_ydata()):
                    rotation = mtransforms.Affine2D().rotate(theta)
                    if axis == 'y':
                        rotation.rotate(-np.pi / 2)
                    ms = mmarkers.MarkerStyle(marker=marker, transform=rotation)
                    self.add_line(mlines.Line2D([theta], [r], marker=ms, **eb_cap_style))
    else:
        for axis in caplines:
            for l in caplines[axis]:
                self.add_line(l)
    self._request_autoscale_view()
    caplines = caplines['x'] + caplines['y']
    errorbar_container = ErrorbarContainer((data_line, tuple(caplines), tuple(barcols)), has_xerr=xerr is not None, has_yerr=yerr is not None, label=label)
    self.containers.append(errorbar_container)
    return errorbar_container