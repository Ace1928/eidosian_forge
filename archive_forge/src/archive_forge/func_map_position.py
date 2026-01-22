from __future__ import annotations
import typing
from contextlib import suppress
import numpy as np
from .._utils import match
from ..exceptions import PlotnineError
from ..iapi import labels_view, layout_details, pos_scales
def map_position(self, layers: Layers):
    """
        Map x & y (position) aesthetics onto the scales.

        e.g If the x scale is scale_x_log10, after this
        function all x, xmax, xmin, ... columns in data
        will be mapped onto log10 scale (log10 transformed).
        The real mapping is handled by the scale.map
        """
    _layout = self.layout
    for layer in layers:
        data = layer.data
        match_id = match(data['PANEL'], _layout['PANEL'])
        if self.panel_scales_x:
            x_vars = list(set(self.panel_scales_x[0].aesthetics) & set(data.columns))
            SCALE_X = _layout['SCALE_X'].iloc[match_id].tolist()
            self.panel_scales_x.map(data, x_vars, SCALE_X)
        if self.panel_scales_y:
            y_vars = list(set(self.panel_scales_y[0].aesthetics) & set(data.columns))
            SCALE_Y = _layout['SCALE_Y'].iloc[match_id].tolist()
            self.panel_scales_y.map(data, y_vars, SCALE_Y)