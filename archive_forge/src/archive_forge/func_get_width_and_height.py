import warnings
from copy import deepcopy
from uuid import uuid4
import bokeh.plotting as bkp
import numpy as np
from bokeh.models import CDSView, ColumnDataSource, GroupFilter, Span
from ....rcparams import rcParams
from ...distplot import plot_dist
from ...kdeplot import plot_kde
from ...plot_utils import (
from .. import show_layout
from . import backend_kwarg_defaults
def get_width_and_height(jointplot, rotate):
    """Compute subplots dimensions for two or more variables."""
    if jointplot:
        if rotate:
            width = int(figsize[0] / (numvars - 1) + 2 * dpi)
            height = int(figsize[1] / (numvars - 1) * dpi)
        else:
            width = int(figsize[0] / (numvars - 1) * dpi)
            height = int(figsize[1] / (numvars - 1) + 2 * dpi)
    else:
        width = int(figsize[0] / (numvars - 1) * dpi)
        height = int(figsize[1] / (numvars - 1) * dpi)
    return (width, height)