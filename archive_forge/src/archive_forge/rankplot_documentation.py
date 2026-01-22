import numpy as np
from bokeh.models import Span
from bokeh.models.annotations import Title
from bokeh.models.tickers import FixedTicker
from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, compute_ranks
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh rank plot.