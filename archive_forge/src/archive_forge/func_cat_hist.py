import numpy as np
from bokeh.models.annotations import Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def cat_hist(val, rug, side, shade, ax, **shade_kwargs):
    """Auxiliary function to plot discrete-violinplots."""
    bins = get_bins(val)
    _, binned_d, _ = histogram(val, bins=bins)
    bin_edges = np.linspace(np.min(val), np.max(val), len(bins))
    heights = np.diff(bin_edges)
    centers = bin_edges[:-1] + heights.mean() / 2
    bar_length = 0.5 * binned_d
    if rug and side == 'both':
        side = 'right'
    if side == 'right':
        left = 0
        right = bar_length
    elif side == 'left':
        left = -bar_length
        right = 0
    elif side == 'both':
        left = -bar_length
        right = bar_length
    ax.hbar(y=centers, left=left, right=right, height=heights, fill_alpha=shade, line_alpha=shade, line_color=None, **shade_kwargs)
    return binned_d