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
def multi_polygons_data(element):
    """
    Expands polygon data which contains holes to a bokeh multi_polygons
    representation. Multi-polygons split by nans are expanded and the
    correct list of holes is assigned to each sub-polygon.
    """
    xs, ys = (element.dimension_values(kd, expanded=False) for kd in element.kdims)
    holes = element.holes()
    xsh, ysh = ([], [])
    for x, y, multi_hole in zip(xs, ys, holes):
        xhs = [[h[:, 0] for h in hole] for hole in multi_hole]
        yhs = [[h[:, 1] for h in hole] for hole in multi_hole]
        array = np.column_stack([x, y])
        splits = np.where(np.isnan(array[:, :2].astype('float')).sum(axis=1))[0]
        arrays = np.split(array, splits + 1) if len(splits) else [array]
        multi_xs, multi_ys = ([], [])
        for i, (path, hx, hy) in enumerate(zip(arrays, xhs, yhs)):
            if i != len(arrays) - 1:
                path = path[:-1]
            multi_xs.append([path[:, 0]] + hx)
            multi_ys.append([path[:, 1]] + hy)
        xsh.append(multi_xs)
        ysh.append(multi_ys)
    return (xsh, ysh)