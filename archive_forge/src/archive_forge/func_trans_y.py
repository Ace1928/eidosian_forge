from __future__ import annotations
import typing
from types import SimpleNamespace as NS
from warnings import warn
from ..exceptions import PlotnineWarning
from ..iapi import panel_ranges, panel_view
from ..positions.position import transform_position
from .coord import coord, dist_euclidean
def trans_y(col: FloatSeries) -> FloatSeries:
    result = transform_value(self.trans_y, col, panel_params.y.range)
    if any(result.isna()):
        warn('Coordinate transform of y aesthetic created one or more NaN values.', PlotnineWarning)
    return result