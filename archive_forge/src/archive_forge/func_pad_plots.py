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
def pad_plots(plots):
    """
    Accepts a grid of bokeh plots in form of a list of lists and
    wraps any DataTable or Tabs in a Column with appropriate
    padding. Required to avoid overlap in gridplot.
    """
    widths = []
    for row in plots:
        row_widths = []
        for p in row:
            width = pad_width(p)
            row_widths.append(width)
        widths.append(row_widths)
    plots = [[Column(p, width=w) if isinstance(p, (DataTable, Tabs)) else p for p, w in zip(row, ws)] for row, ws in zip(plots, widths)]
    return plots