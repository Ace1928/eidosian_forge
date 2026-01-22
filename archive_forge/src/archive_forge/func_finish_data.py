from __future__ import annotations
import typing
from contextlib import suppress
import numpy as np
from .._utils import match
from ..exceptions import PlotnineError
from ..iapi import labels_view, layout_details, pos_scales
def finish_data(self, layers: Layers):
    """
        Modify data before it is drawn out by the geom

        Parameters
        ----------
        layers : list
            List of layers
        """
    for layer in layers:
        layer.data = self.facet.finish_data(layer.data, self)