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
def test_draw_line_vertical_horizontal():
    x0, y0 = (3, 3)
    x1, y1 = (3, 0)
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    out = new_agg()
    out[:4, 3] = 1
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(y0, x0, y1, x1, 0, True, agg)
    out = new_agg()
    out[3, :4] = 1
    np.testing.assert_equal(agg, out)