from __future__ import annotations
import typing
from contextlib import suppress
import numpy as np
from .._utils import match
from ..exceptions import PlotnineError
from ..iapi import labels_view, layout_details, pos_scales
def train_position(self, layers: Layers, scales: Scales):
    """
        Create all the required x & y panel_scales

        And set the ranges for each scale according to the data

        Notes
        -----
        The number of x or y scales depends on the facetting,
        particularly the scales parameter. e.g if `scales="free"`{.py}
        then each panel will have separate x and y scales, and
        if `scales="fixed"`{.py} then all panels will share an x
        scale and a y scale.
        """
    layout = self.layout
    if not hasattr(self, 'panel_scales_x') and scales.x:
        result = self.facet.init_scales(layout, scales.x, None)
        self.panel_scales_x = result.x
    if not hasattr(self, 'panel_scales_y') and scales.y:
        result = self.facet.init_scales(layout, None, scales.y)
        self.panel_scales_y = result.y
    self.facet.train_position_scales(self, layers)