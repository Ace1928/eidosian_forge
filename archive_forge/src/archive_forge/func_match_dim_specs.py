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
def match_dim_specs(specs1, specs2):
    """Matches dimension specs used to link axes.

    Axis dimension specs consists of a list of tuples corresponding
    to each dimension, each tuple spec has the form (name, label, unit).
    The name and label must match exactly while the unit only has to
    match if both specs define one.
    """
    if (specs1 is None or specs2 is None) or len(specs1) != len(specs2):
        return False
    for spec1, spec2 in zip(specs1, specs2):
        for s1, s2 in zip(spec1, spec2):
            if s1 is None or s2 is None:
                continue
            if s1 != s2:
                return False
    return True