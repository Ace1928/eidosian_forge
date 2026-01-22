from numbers import Number
from typing import Optional
import numpy as np
from bokeh.models.annotations import Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram
from ...kdeplot import plot_kde
from ...plot_utils import (
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def plot_posterior(ax, length_plotters, rows, cols, figsize, plotters, bw, circular, bins, kind, point_estimate, round_to, hdi_prob, multimodal, skipna, textsize, ref_val, rope, ref_val_color, rope_color, labeller, kwargs, backend_kwargs, show):
    """Bokeh posterior plot."""
    if backend_kwargs is None:
        backend_kwargs = {}
    backend_kwargs = {**backend_kwarg_defaults(('dpi', 'plot.bokeh.figure.dpi')), **backend_kwargs}
    figsize, ax_labelsize, *_, linewidth, _ = _scale_fig_size(figsize, textsize, rows, cols)
    if ax is None:
        ax = create_axes_grid(length_plotters, rows, cols, figsize=figsize, backend_kwargs=backend_kwargs)
    else:
        ax = np.atleast_2d(ax)
    idx = 0
    for (var_name, selection, isel, x), ax_ in zip(plotters, (item for item in ax.flatten() if item is not None)):
        _plot_posterior_op(idx, x.flatten(), var_name, selection, ax=ax_, bw=bw, circular=circular, bins=bins, kind=kind, point_estimate=point_estimate, round_to=round_to, hdi_prob=hdi_prob, multimodal=multimodal, skipna=skipna, linewidth=linewidth, ref_val=ref_val, rope=rope, ref_val_color=ref_val_color, rope_color=rope_color, ax_labelsize=ax_labelsize, **kwargs)
        idx += 1
        _title = Title()
        _title.text = labeller.make_label_vert(var_name, selection, isel)
        ax_.title = _title
    show_layout(ax, show)
    return ax