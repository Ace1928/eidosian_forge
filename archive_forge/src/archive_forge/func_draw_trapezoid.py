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
def draw_trapezoid(x0, x1, y0, y1, y2, y3, i, trapezoid_start, stacked, agg):
    """
    Helper to draw line with fixed bounds and scale values.
    """
    sx, tx, sy, ty = (1, 0, 1, 0)
    xmin, xmax, ymin, ymax = (0, 5, 0, 5)
    _draw_trapezoid(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, agg)