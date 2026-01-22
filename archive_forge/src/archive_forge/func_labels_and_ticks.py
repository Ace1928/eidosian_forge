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
def labels_and_ticks(self):
    """Collect labels and ticks from plotters."""
    val = self.plotters.values()

    def label_idxs():
        labels, idxs = ([], [])
        for plotter in val:
            sub_labels, sub_idxs, _, _, _ = plotter.labels_ticks_and_vals()
            labels_to_idxs = defaultdict(list)
            for label, idx in zip(sub_labels, sub_idxs):
                labels_to_idxs[label].append(idx)
            sub_idxs = []
            sub_labels = []
            for label, all_idx in labels_to_idxs.items():
                sub_labels.append(label)
                sub_idxs.append(np.mean([j for j in all_idx]))
            labels.append(sub_labels)
            idxs.append(sub_idxs)
        return (np.concatenate(labels), np.concatenate(idxs))
    return label_idxs()