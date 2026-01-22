from __future__ import annotations
import logging # isort:skip
import sys
from collections.abc import Iterable
import numpy as np
from ..core.properties import ColorSpec
from ..models import ColumnarDataSource, ColumnDataSource, GlyphRenderer
from ..util.strings import nice_join
from ._legends import pop_legend_kwarg, update_legend
def make_glyph(glyphclass, kws, extra):
    if extra is None:
        return None
    kws = kws.copy()
    kws.update(extra)
    return glyphclass(**kws)