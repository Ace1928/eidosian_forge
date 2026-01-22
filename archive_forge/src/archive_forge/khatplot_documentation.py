from collections.abc import Iterable
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import Span
from matplotlib.colors import to_rgba_array
from ....stats.density_utils import histogram
from ...plot_utils import _scale_fig_size, color_from_dim, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
Bokeh khat plot.