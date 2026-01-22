from __future__ import annotations
import logging # isort:skip
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from ..core.properties import Datetime
from ..core.property.singletons import Intrinsic
from ..models import (
def process_axis_and_grid(plot: Plot, axis_type: AxisType | None, axis_location: AxisLocation | None, minor_ticks: int | Literal['auto'] | None, axis_label: str | BaseText | None, rng: Range, dim: Dim) -> None:
    axiscls, axiskw = _get_axis_class(axis_type, rng, dim)
    if axiscls:
        axis = axiscls(**axiskw)
        if isinstance(axis.ticker, ContinuousTicker):
            axis.ticker.num_minor_ticks = _get_num_minor_ticks(axiscls, minor_ticks)
        if axis_label:
            axis.axis_label = axis_label
        grid = Grid(dimension=dim, axis=axis)
        plot.add_layout(grid, 'center')
        if axis_location is not None:
            getattr(plot, axis_location).append(axis)