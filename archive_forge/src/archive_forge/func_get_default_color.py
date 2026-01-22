from __future__ import annotations
import logging # isort:skip
import sys
from collections.abc import Iterable
import numpy as np
from ..core.properties import ColorSpec
from ..models import ColumnarDataSource, ColumnDataSource, GlyphRenderer
from ..util.strings import nice_join
from ._legends import pop_legend_kwarg, update_legend
def get_default_color(plot=None):
    colors = ['#1f77b4', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    if plot:
        renderers = plot.renderers
        renderers = [x for x in renderers if x.__view_model__ == 'GlyphRenderer']
        num_renderers = len(renderers)
        return colors[num_renderers]
    else:
        return colors[0]