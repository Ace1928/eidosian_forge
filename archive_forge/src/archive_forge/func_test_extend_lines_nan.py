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
def test_extend_lines_nan():
    xs = np.array([-3, -2, np.nan, 0, 1])
    ys = np.array([-3, -2, np.nan, 0, 1])
    agg = new_agg()
    sx, tx, sy, ty = vt
    xmin, xmax, ymin, ymax = bounds
    buffer = np.empty(0)
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    out = np.diag([1, 1, 0, 1, 0])
    np.testing.assert_equal(agg, out)