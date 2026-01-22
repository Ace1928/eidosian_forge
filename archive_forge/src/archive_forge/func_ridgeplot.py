from collections import OrderedDict, defaultdict
from itertools import cycle, tee
import bokeh.plotting as bkp
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import Band, ColumnDataSource, DataRange1d
from bokeh.models.annotations import Title, Legend
from bokeh.models.tickers import FixedTicker
from ....sel_utils import xarray_var_iter
from ....rcparams import rcParams
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ....stats.diagnostics import _ess, _rhat
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults
def ridgeplot(self, hdi_prob, mult, ridgeplot_kind):
    """Get data for each ridgeplot for the variable."""
    xvals, hdi_vals, yvals, pdfs, pdfs_q, colors, model_names = ([], [], [], [], [], [], [])
    for y, _, model_name, *_, values, color in self.iterator():
        yvals.append(y)
        colors.append(color)
        model_names.append(model_name)
        values = values.flatten()
        values = values[np.isfinite(values)]
        if hdi_prob != 1:
            hdi_ = hdi(values, hdi_prob, multimodal=False)
        else:
            hdi_ = (min(values), max(values))
        if ridgeplot_kind == 'auto':
            kind = 'hist' if np.all(np.mod(values, 1) == 0) else 'density'
        else:
            kind = ridgeplot_kind
        if kind == 'hist':
            bins = get_bins(values)
            _, density, x = histogram(values, bins=bins)
            x = x[:-1]
        elif kind == 'density':
            x, density = kde(values)
        density_q = density.cumsum() / density.sum()
        xvals.append(x)
        pdfs.append(density)
        pdfs_q.append(density_q)
        hdi_vals.append(hdi_)
    scaling = max((np.max(j) for j in pdfs))
    for y, x, hdi_val, pdf, pdf_q, color, model_name in zip(yvals, xvals, hdi_vals, pdfs, pdfs_q, colors, model_names):
        yield (x, y, mult * pdf / scaling + y, hdi_val, pdf_q, color, model_name)