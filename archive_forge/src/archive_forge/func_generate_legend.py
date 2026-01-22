from .. import utils
from .._lazyload import matplotlib as mpl
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import temp_fontsize
import numpy as np
import warnings
@utils._with_pkg(pkg='matplotlib', min_version=3)
def generate_legend(cmap, ax, title=None, marker='o', markersize=10, loc='best', bbox_to_anchor=None, fontsize=None, title_fontsize=None, max_rows=10, ncol=None, **kwargs):
    """Generate a legend on an axis.

    Parameters
    ----------
    cmap : dict
        Dictionary of label-color pairs.
    ax : `matplotlib.axes.Axes`
        Axis on which to draw the legend
    title : str, optional (default: None)
        Title to display alongside colorbar
    marker : str, optional (default: 'o')
        `matplotlib` marker to use for legend points
    markersize : float, optional (default: 10)
        Size of legend points
    loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    bbox_to_anchor : `BboxBase`, 2-tuple, or 4-tuple
        Box that is used to position the legend in conjunction with loc.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    fontsize : int, optional (default: None)
        Font size for legend labels
    title_fontsize : int, optional (default: None)
        Font size for legend title
    max_rows : int, optional (default: 10)
        Maximum number of labels in a column before overflowing to
        multi-column legend
    ncol : int, optional (default: None)
        Number of legend columns. Overrides `max_rows`.
    kwargs : additional arguments for `plt.legend`

    Returns
    -------
    legend : `matplotlib.legend.Legend`
    """
    fontsize = parse_fontsize(fontsize, 'large')
    title_fontsize = parse_fontsize(title_fontsize, 'x-large')
    handles = [mpl.lines.Line2D([], [], marker=marker, color=color, linewidth=0, label=label, markersize=markersize) for label, color in cmap.items()]
    if ncol is None:
        ncol = max(1, np.ceil(len(cmap) / max_rows).astype(int))
    legend = ax.legend(handles=handles, title=title, loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, ncol=ncol, **kwargs)
    plt.setp(legend.get_title(), fontsize=title_fontsize)
    return legend