from __future__ import annotations
import pandas as pd
import numpy as np
import pytest
from datashader.datashape import dshape
from datashader.glyphs import Point, LinesAxis1, Glyph
from datashader.glyphs.area import _build_draw_trapezoid_y
from datashader.glyphs.line import (
from datashader.glyphs.trimesh import(
from datashader.utils import ngjit
def test_draw_line_same_point():
    x0, y0 = (4, 4)
    x1, y1 = (4, 4)
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    assert agg.sum() == 1
    assert agg[4, 4] == 1
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, False, agg)
    assert agg.sum() == 1
    assert agg[4, 4] == 1
    x0, y0 = (4, 4)
    x1, y1 = (10, 10)
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    assert agg.sum() == 1
    assert agg[4, 4] == 1
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, False, agg)
    assert agg.sum() == 0
    assert agg[4, 4] == 0