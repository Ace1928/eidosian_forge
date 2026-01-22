from .. import select
from .. import utils
from .._lazyload import matplotlib as mpl
from . import colors
from .tools import create_colormap
from .tools import create_normalize
from .tools import generate_colorbar
from .tools import generate_legend
from .tools import label_axis
from .utils import _get_figure
from .utils import _in_ipynb
from .utils import _is_color_array
from .utils import _with_default
from .utils import parse_fontsize
from .utils import show
from .utils import temp_fontsize
import numbers
import numpy as np
import pandas as pd
import warnings
@utils._with_pkg(pkg='matplotlib', min_version=3)
def scatter3d(data, c=None, cmap=None, cmap_scale='linear', s=None, mask=None, discrete=None, ax=None, legend=None, colorbar=None, shuffle=True, figsize=None, ticks=True, xticks=None, yticks=None, zticks=None, ticklabels=True, xticklabels=None, yticklabels=None, zticklabels=None, label_prefix=None, xlabel=None, ylabel=None, zlabel=None, title=None, fontsize=None, legend_title=None, legend_loc='best', legend_anchor=None, legend_ncol=None, elev=None, azim=None, filename=None, dpi=None, **plot_kwargs):
    """Create a 3D scatter plot.

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Only the first two components will be used.
    c : list-like or None, optional (default: None)
        Color vector. Can be a single color value (RGB, RGBA, or named
        matplotlib colors), an array of these of length n_samples, or a list of
        discrete or continuous values of any data type. If `c` is not a single
        or list of matplotlib colors, the values in `c` will be used to
        populate the legend / colorbar with colors from `cmap`
    cmap : `matplotlib` colormap, str, dict, list or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a list, expects one color for every
        unique value in `c`, otherwise interpolates between given colors for
        continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    cmap_scale : {'linear', 'log', 'symlog', 'sqrt'} or `matplotlib.colors.Normalize`,
        optional (default: 'linear')
        Colormap normalization scale. For advanced use, see
        <https://matplotlib.org/users/colormapnorms.html>
    s : float, optional (default: None)
        Point size. If `None`, set to 200 / sqrt(n_samples)
    mask : list-like, optional (default: None)
        boolean mask to hide data points
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: None)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar. If `None`, a legend is created where possible.
    colorbar : bool, optional (default: None)
        Synonym for `legend`
    shuffle : bool, optional (default: True)
        If True. shuffles the order of points on the plot.
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    ticks : True, False, or list-like (default: True)
        If True, keeps default axis ticks. If False, removes axis ticks.
        If a list, sets custom axis ticks
    {x,y,z}ticks : True, False, or list-like (default: None)
        If set, overrides `ticks`
    ticklabels : True, False, or list-like (default: True)
        If True, keeps default axis tick labels. If False, removes axis tick labels.
        If a list, sets custom axis tick labels
    {x,y,z}ticklabels : True, False, or list-like (default: None)
        If set, overrides `ticklabels`
    label_prefix : str or None (default: None)
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    {x,y,z}label : str or None (default : None)
        Axis labels. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set
        unless the data is a pandas Series, in which case the series name is used.
        Override this behavior with `{x,y,z}label=False`
    title : str or None (default: None)
        axis title. If None, no title is set.
    fontsize : float or None (default: None)
        Base font size.
    legend_title : str (default: None)
        title for the colorbar of legend
    legend_loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location. Only used for discrete data.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    legend_anchor : `BboxBase`, 2-tuple, or 4-tuple
        Box that is used to position the legend in conjunction with loc.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    legend_ncol : `int` or `None`, optimal (default: None)
        Number of columns to show in the legend.
        If None, defaults to a maximum of entries per column.
    vmin, vmax : float, optional (default: None)
        Range of values to use as the range for the colormap.
        Only used if data is continuous
    elev : int, optional (default: None)
        Elevation angle of viewpoint from horizontal, in degrees
    azim : int, optional (default: None)
        Azimuth angle in x-y plane of viewpoint
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **plot_kwargs : keyword arguments
        Extra arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

    Examples
    --------
    >>> import scprep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.normal(0, 1, [200, 3])
    >>> # Continuous color vector
    >>> colors = data[:, 0]
    >>> scprep.plot.scatter3d(data, c=colors)
    >>> # Discrete color vector with custom colormap
    >>> colors = np.random.choice(['a','b'], data.shape[0], replace=True)
    >>> data[colors == 'a'] += 5
    >>> scprep.plot.scatter3d(
            data, c=colors, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'}
        )
    """
    if isinstance(data, list):
        data = utils.toarray(data)
    if isinstance(data, np.ndarray):
        data = np.atleast_2d(data)
    try:
        x = select.select_cols(data, idx=0)
        y = select.select_cols(data, idx=1)
        z = select.select_cols(data, idx=2)
    except IndexError:
        raise ValueError('Expected data.shape[1] >= 3. Got {}'.format(data.shape[1]))
    return scatter(x=x, y=y, z=z, c=c, cmap=cmap, cmap_scale=cmap_scale, s=s, mask=mask, discrete=discrete, ax=ax, legend=legend, colorbar=colorbar, shuffle=shuffle, figsize=figsize, ticks=ticks, xticks=xticks, yticks=yticks, zticks=zticks, ticklabels=ticklabels, xticklabels=xticklabels, yticklabels=yticklabels, zticklabels=zticklabels, label_prefix=label_prefix, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title, fontsize=fontsize, legend_title=legend_title, legend_loc=legend_loc, legend_anchor=legend_anchor, elev=elev, azim=azim, filename=filename, dpi=dpi, **plot_kwargs)