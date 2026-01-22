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
def test_draw_line():
    x0, y0 = (0, 0)
    x1, y1 = (3, 3)
    out = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x1, y1, x0, y0, 0, True, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, False, agg)
    out[0, 0] = 0
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x1, y1, x0, y0, 0, False, agg)
    out[0, 0] = 1
    out[3, 3] = 0
    np.testing.assert_equal(agg, out)
    x0, y0 = (0, 4)
    x1, y1 = (3, 1)
    out = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, True, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x1, y1, x0, y0, 0, True, agg)
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x0, y0, x1, y1, 0, False, agg)
    out[4, 0] = 0
    np.testing.assert_equal(agg, out)
    agg = new_agg()
    draw_segment(x1, y1, x0, y0, 0, False, agg)
    out[4, 0] = 1
    out[1, 3] = 0