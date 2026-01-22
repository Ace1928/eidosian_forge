import calendar
import datetime as dt
import re
import time
from collections import defaultdict
from contextlib import contextmanager, suppress
from itertools import permutations
import bokeh
import numpy as np
import pandas as pd
from bokeh.core.json_encoder import serialize_json  # noqa (API import)
from bokeh.core.property.datetime import Datetime
from bokeh.core.validation import silence
from bokeh.layouts import Column, Row, group_tools
from bokeh.models import (
from bokeh.models.formatters import PrintfTickFormatter, TickFormatter
from bokeh.models.scales import CategoricalScale, LinearScale, LogScale
from bokeh.models.widgets import DataTable, Div
from bokeh.plotting import figure
from bokeh.themes import built_in_themes
from bokeh.themes.theme import Theme
from packaging.version import Version
from ...core.layout import Layout
from ...core.ndmapping import NdMapping
from ...core.overlay import NdOverlay, Overlay
from ...core.spaces import DynamicMap, get_nested_dmaps
from ...core.util import (
from ...util.warnings import warn
from ..util import dim_axis_label
def compute_plot_size(plot):
    """
    Computes the size of bokeh models that make up a layout such as
    figures, rows, columns, and Plot.
    """
    if isinstance(plot, (GridBox, GridPlot)):
        ndmapping = NdMapping({(x, y): fig for fig, y, x in plot.children}, kdims=['x', 'y'])
        cols = ndmapping.groupby('x')
        rows = ndmapping.groupby('y')
        width = sum([max([compute_plot_size(f)[0] for f in col]) for col in cols])
        height = sum([max([compute_plot_size(f)[1] for f in row]) for row in rows])
        return (width, height)
    elif isinstance(plot, (Div, Toolbar)):
        return (0, 0)
    elif isinstance(plot, (Row, Column, Tabs)):
        if not plot.children:
            return (0, 0)
        if isinstance(plot, Row) or (isinstance(plot, Toolbar) and plot.toolbar_location not in ['right', 'left']):
            w_agg, h_agg = (np.sum, np.max)
        elif isinstance(plot, Tabs):
            w_agg, h_agg = (np.max, np.max)
        else:
            w_agg, h_agg = (np.max, np.sum)
        widths, heights = zip(*[compute_plot_size(child) for child in plot.children])
        return (w_agg(widths), h_agg(heights))
    elif isinstance(plot, figure):
        if plot.width:
            width = plot.width
        else:
            width = plot.frame_width + plot.min_border_right + plot.min_border_left
        if plot.height:
            height = plot.height
        else:
            height = plot.frame_height + plot.min_border_bottom + plot.min_border_top
        return (width, height)
    elif isinstance(plot, (Plot, DataTable, Spacer)):
        return (plot.width, plot.height)
    else:
        return (0, 0)