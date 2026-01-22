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
def test_extend_lines_all_out_of_bounds():
    xs = np.array([-100, -200, -100])
    ys = np.array([0, 0, 1])
    agg = new_agg()
    sx, tx, sy, ty = vt
    xmin, xmax, ymin, ymax = bounds
    buffer = np.empty(0)
    extend_line(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, True, buffer, agg)
    assert agg.sum() == 0