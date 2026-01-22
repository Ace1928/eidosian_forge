from __future__ import annotations
import logging # isort:skip
import numpy as np
from ..core.properties import field, value
from ..models import Legend, LegendItem
from ..util.strings import nice_join
def update_legend(plot, legend_kwarg, glyph_renderer):
    legend = _get_or_create_legend(plot)
    kwarg, value = next(iter(legend_kwarg.items()))
    _LEGEND_KWARG_HANDLERS[kwarg](value, legend, glyph_renderer)