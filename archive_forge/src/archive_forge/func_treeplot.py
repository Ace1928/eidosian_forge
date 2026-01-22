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
def treeplot(self, qlist, hdi_prob):
    """Get data for each treeplot for the variable."""
    for y, _, model_name, _, selection, values, color in self.iterator():
        ntiles = np.percentile(values.flatten(), qlist)
        ntiles[0], ntiles[-1] = hdi(values.flatten(), hdi_prob, multimodal=False)
        yield (y, model_name, selection, ntiles, color)