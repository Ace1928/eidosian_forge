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
def test_extend_lines():
    xs = np.array([0, -2, -2, 0, 0])
    ys = np.array([-1, -1, 1.1, 1.1, -1])
    out = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]])
    agg = new_agg()
    sx, tx, sy, ty = vt
    xmin, xmax, ymin, ymax = bounds
    buffer = np.empty(0)
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, False, buffer, agg)
    np.testing.assert_equal(agg, out)
    out[2, 3] += 1
    agg = new_agg()
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    np.testing.assert_equal(agg, out)
    xs = np.array([2, 1, 0, -1, -4, -1, -100, -1, 2])
    ys = np.array([-1, -2, -3, -4, -1, 2, 100, 2, -1])
    out = np.array([[0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, 1, 0, 1, 0], [0, 0, 0, 0, 0]])
    agg = new_agg()
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    np.testing.assert_equal(agg, out)