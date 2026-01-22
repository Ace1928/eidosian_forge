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
def test_extend_lines_exact_bounds():
    xs = np.array([-3, 1, 1, -3, -3])
    ys = np.array([-3, -3, 1, 1, -3])
    agg = np.zeros((4, 4), dtype='i4')
    sx, tx, sy, ty = vt
    xmin, xmax, ymin, ymax = bounds
    buffer = np.empty(0)
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    out = np.array([[2, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])
    np.testing.assert_equal(agg, out)
    agg = np.zeros((4, 4), dtype='i4')
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, False, buffer, agg)
    out = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])
    np.testing.assert_equal(agg, out)