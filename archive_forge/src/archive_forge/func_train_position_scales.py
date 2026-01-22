from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from .._utils import cross_join, match
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from .strips import Strips
def train_position_scales(self, layout: Layout, layers: Layers) -> facet:
    """
        Compute ranges for the x and y scales
        """
    _layout = layout.layout
    panel_scales_x = layout.panel_scales_x
    panel_scales_y = layout.panel_scales_y
    for layer in layers:
        data = layer.data
        match_id = match(data['PANEL'], _layout['PANEL'])
        if panel_scales_x:
            x_vars = list(set(panel_scales_x[0].aesthetics) & set(data.columns))
            SCALE_X = _layout['SCALE_X'].iloc[match_id].tolist()
            panel_scales_x.train(data, x_vars, SCALE_X)
        if panel_scales_y:
            y_vars = list(set(panel_scales_y[0].aesthetics) & set(data.columns))
            SCALE_Y = _layout['SCALE_Y'].iloc[match_id].tolist()
            panel_scales_y.train(data, y_vars, SCALE_Y)
    return self