import numpy as np
from bokeh.models import DataRange1d
from bokeh.models.tickers import FixedTicker
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh parallel plot.