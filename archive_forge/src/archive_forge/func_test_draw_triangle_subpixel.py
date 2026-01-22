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
def test_draw_triangle_subpixel():
    """Assert that we draw subpixel triangles properly, both with and without
    interpolation.
    """
    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3))
    out = np.array([[0, 0, 5.5, 0, 0], [0, 4.9, 4.5, 4.1, 0], [4.3, 3.9, 3.5, 3.1, 2.7], [0, 0, 8, 0, 0]])
    agg = np.zeros((4, 5), dtype='f4')
    draw_triangle_interp(tri[:3], (0, 4, 0, 5), (0, 0, 0), (agg,), (6, 4, 2))
    draw_triangle_interp(tri[3:6], (2, 2, 3, 3), (0, 0, 0), (agg,), (6, 4, 2))
    draw_triangle_interp(tri[6:], (2, 2, 3, 3), (0, 0, 0), (agg,), (6, 4, 2))
    np.testing.assert_allclose(agg, out)
    out = np.array([[0, 0, 2, 0, 0], [0, 2, 2, 2, 0], [2, 2, 2, 2, 2], [0, 0, 4, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri[:3], (0, 4, 0, 5), (0, 0, 0), (agg,), 2)
    draw_triangle(tri[3:6], (2, 2, 3, 3), (0, 0, 0), (agg,), 2)
    draw_triangle(tri[6:], (2, 2, 3, 3), (0, 0, 0), (agg,), 2)
    np.testing.assert_equal(agg, out)