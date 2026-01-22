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
@_preprocess_data()
def boxplot(self, x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True, autorange=False, zorder=None, capwidths=None):
    """
        Draw a box and whisker plot.

        The box extends from the first quartile (Q1) to the third
        quartile (Q3) of the data, with a line at the median.
        The whiskers extend from the box to the farthest data point
        lying within 1.5x the inter-quartile range (IQR) from the box.
        Flier points are those past the end of the whiskers.
        See https://en.wikipedia.org/wiki/Box_plot for reference.

        .. code-block:: none

                  Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR
                               |-----:-----|
               o      |--------|     :     |--------|    o  o
                               |-----:-----|
             flier             <----------->            fliers
                                    IQR


        Parameters
        ----------
        x : Array or a sequence of vectors.
            The input data.  If a 2D array, a boxplot is drawn for each column
            in *x*.  If a sequence of 1D arrays, a boxplot is drawn for each
            array in *x*.

        notch : bool, default: False
            Whether to draw a notched boxplot (`True`), or a rectangular
            boxplot (`False`).  The notches represent the confidence interval
            (CI) around the median.  The documentation for *bootstrap*
            describes how the locations of the notches are computed by
            default, but their locations may also be overridden by setting the
            *conf_intervals* parameter.

            .. note::

                In cases where the values of the CI are less than the
                lower quartile or greater than the upper quartile, the
                notches will extend beyond the box, giving it a
                distinctive "flipped" appearance. This is expected
                behavior and consistent with other statistical
                visualization packages.

        sym : str, optional
            The default symbol for flier points.  An empty string ('') hides
            the fliers.  If `None`, then the fliers default to 'b+'.  More
            control is provided by the *flierprops* parameter.

        vert : bool, default: True
            If `True`, draws vertical boxes.
            If `False`, draw horizontal boxes.

        whis : float or (float, float), default: 1.5
            The position of the whiskers.

            If a float, the lower whisker is at the lowest datum above
            ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest datum
            below ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the first and
            third quartiles.  The default value of ``whis = 1.5`` corresponds
            to Tukey's original definition of boxplots.

            If a pair of floats, they indicate the percentiles at which to
            draw the whiskers (e.g., (5, 95)).  In particular, setting this to
            (0, 100) results in whiskers covering the whole range of the data.

            In the edge case where ``Q1 == Q3``, *whis* is automatically set
            to (0, 100) (cover the whole range of the data) if *autorange* is
            True.

            Beyond the whiskers, data are considered outliers and are plotted
            as individual points.

        bootstrap : int, optional
            Specifies whether to bootstrap the confidence intervals
            around the median for notched boxplots. If *bootstrap* is
            None, no bootstrapping is performed, and notches are
            calculated using a Gaussian-based asymptotic approximation
            (see McGill, R., Tukey, J.W., and Larsen, W.A., 1978, and
            Kendall and Stuart, 1967). Otherwise, bootstrap specifies
            the number of times to bootstrap the median to determine its
            95% confidence intervals. Values between 1000 and 10000 are
            recommended.

        usermedians : 1D array-like, optional
            A 1D array-like of length ``len(x)``.  Each entry that is not
            `None` forces the value of the median for the corresponding
            dataset.  For entries that are `None`, the medians are computed
            by Matplotlib as normal.

        conf_intervals : array-like, optional
            A 2D array-like of shape ``(len(x), 2)``.  Each entry that is not
            None forces the location of the corresponding notch (which is
            only drawn if *notch* is `True`).  For entries that are `None`,
            the notches are computed by the method specified by the other
            parameters (e.g., *bootstrap*).

        positions : array-like, optional
            The positions of the boxes. The ticks and limits are
            automatically set to match the positions. Defaults to
            ``range(1, N+1)`` where N is the number of boxes to be drawn.

        widths : float or array-like
            The widths of the boxes.  The default is 0.5, or ``0.15*(distance
            between extreme positions)``, if that is smaller.

        patch_artist : bool, default: False
            If `False` produces boxes with the Line2D artist. Otherwise,
            boxes are drawn with Patch artists.

        labels : sequence, optional
            Labels for each dataset (one per dataset).

        manage_ticks : bool, default: True
            If True, the tick locations and labels will be adjusted to match
            the boxplot positions.

        autorange : bool, default: False
            When `True` and the data are distributed such that the 25th and
            75th percentiles are equal, *whis* is set to (0, 100) such
            that the whisker ends are at the minimum and maximum of the data.

        meanline : bool, default: False
            If `True` (and *showmeans* is `True`), will try to render the
            mean as a line spanning the full width of the box according to
            *meanprops* (see below).  Not recommended if *shownotches* is also
            True.  Otherwise, means will be shown as points.

        zorder : float, default: ``Line2D.zorder = 2``
            The zorder of the boxplot.

        Returns
        -------
        dict
          A dictionary mapping each component of the boxplot to a list
          of the `.Line2D` instances created. That dictionary has the
          following keys (assuming vertical boxplots):

          - ``boxes``: the main body of the boxplot showing the
            quartiles and the median's confidence intervals if
            enabled.

          - ``medians``: horizontal lines at the median of each box.

          - ``whiskers``: the vertical lines extending to the most
            extreme, non-outlier data points.

          - ``caps``: the horizontal lines at the ends of the
            whiskers.

          - ``fliers``: points representing data that extend beyond
            the whiskers (fliers).

          - ``means``: points or lines representing the means.

        Other Parameters
        ----------------
        showcaps : bool, default: True
            Show the caps on the ends of whiskers.
        showbox : bool, default: True
            Show the central box.
        showfliers : bool, default: True
            Show the outliers beyond the caps.
        showmeans : bool, default: False
            Show the arithmetic means.
        capprops : dict, default: None
            The style of the caps.
        capwidths : float or array, default: None
            The widths of the caps.
        boxprops : dict, default: None
            The style of the box.
        whiskerprops : dict, default: None
            The style of the whiskers.
        flierprops : dict, default: None
            The style of the fliers.
        medianprops : dict, default: None
            The style of the median.
        meanprops : dict, default: None
            The style of the mean.
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        See Also
        --------
        violinplot : Draw an estimate of the probability density function.
        """
    if whis is None:
        whis = mpl.rcParams['boxplot.whiskers']
    if bootstrap is None:
        bootstrap = mpl.rcParams['boxplot.bootstrap']
    bxpstats = cbook.boxplot_stats(x, whis=whis, bootstrap=bootstrap, labels=labels, autorange=autorange)
    if notch is None:
        notch = mpl.rcParams['boxplot.notch']
    if vert is None:
        vert = mpl.rcParams['boxplot.vertical']
    if patch_artist is None:
        patch_artist = mpl.rcParams['boxplot.patchartist']
    if meanline is None:
        meanline = mpl.rcParams['boxplot.meanline']
    if showmeans is None:
        showmeans = mpl.rcParams['boxplot.showmeans']
    if showcaps is None:
        showcaps = mpl.rcParams['boxplot.showcaps']
    if showbox is None:
        showbox = mpl.rcParams['boxplot.showbox']
    if showfliers is None:
        showfliers = mpl.rcParams['boxplot.showfliers']
    if boxprops is None:
        boxprops = {}
    if whiskerprops is None:
        whiskerprops = {}
    if capprops is None:
        capprops = {}
    if medianprops is None:
        medianprops = {}
    if meanprops is None:
        meanprops = {}
    if flierprops is None:
        flierprops = {}
    if patch_artist:
        boxprops['linestyle'] = 'solid'
        if 'color' in boxprops:
            boxprops['edgecolor'] = boxprops.pop('color')
    if sym is not None:
        if sym == '':
            flierprops = dict(linestyle='none', marker='', color='none')
            showfliers = False
        else:
            _, marker, color = _process_plot_format(sym)
            if marker is not None:
                flierprops['marker'] = marker
            if color is not None:
                flierprops['color'] = color
                flierprops['markerfacecolor'] = color
                flierprops['markeredgecolor'] = color
    if usermedians is not None:
        if len(np.ravel(usermedians)) != len(bxpstats) or np.shape(usermedians)[0] != len(bxpstats):
            raise ValueError("'usermedians' and 'x' have different lengths")
        else:
            for stats, med in zip(bxpstats, usermedians):
                if med is not None:
                    stats['med'] = med
    if conf_intervals is not None:
        if len(conf_intervals) != len(bxpstats):
            raise ValueError("'conf_intervals' and 'x' have different lengths")
        else:
            for stats, ci in zip(bxpstats, conf_intervals):
                if ci is not None:
                    if len(ci) != 2:
                        raise ValueError('each confidence interval must have two values')
                    else:
                        if ci[0] is not None:
                            stats['cilo'] = ci[0]
                        if ci[1] is not None:
                            stats['cihi'] = ci[1]
    artists = self.bxp(bxpstats, positions=positions, widths=widths, vert=vert, patch_artist=patch_artist, shownotches=notch, showmeans=showmeans, showcaps=showcaps, showbox=showbox, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, meanprops=meanprops, meanline=meanline, showfliers=showfliers, capprops=capprops, whiskerprops=whiskerprops, manage_ticks=manage_ticks, zorder=zorder, capwidths=capwidths)
    return artists