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
def cds_column_replace(source, data):
    """
    Determine if the CDS.data requires a full replacement or simply
    needs to be updated. A replacement is required if untouched
    columns are not the same length as the columns being updated.
    """
    current_length = [len(v) for v in source.data.values() if isinstance(v, (list,) + arraylike_types)]
    new_length = [len(v) for v in data.values() if isinstance(v, (list, np.ndarray))]
    untouched = [k for k in source.data if k not in data]
    return bool(untouched and current_length and new_length and (current_length[0] != new_length[0]))