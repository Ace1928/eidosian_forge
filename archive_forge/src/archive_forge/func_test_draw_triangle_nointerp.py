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
def test_draw_triangle_nointerp():
    """Assert that we draw triangles properly, without interpolation enabled.
    """
    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5))
    out = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 4, 0, 3), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)
    tri = ((2.4, -0.5), (-0.5, 2.4), (2.4, 2.4))
    out = np.array([[0, 0, 2, 0, 0], [0, 2, 2, 0, 0], [2, 2, 2, 0, 0], [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 4, 0, 3), (0, 0, 0), (agg,), 2)
    np.testing.assert_equal(agg, out)
    tri = ((2.4, -0.5), (-0.5, 2.4), (2.4, 2.4), (2.4, -0.5), (2.4, 3.5), (4.5, -0.5))
    out = np.array([[0, 0, 3, 4, 4], [0, 3, 3, 4, 0], [3, 3, 3, 4, 0], [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri[:3], (0, 4, 0, 3), (0, 0, 0), (agg,), 3)
    draw_triangle(tri[3:], (0, 4, 0, 3), (0, 0, 0), (agg,), 4)
    np.testing.assert_equal(agg, out)
    tri = ((2, -0.5), (-0.5, 2.5), (4.5, 2.5))
    out = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (0, 3, 0, 2), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)
    out = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 0, 2), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)
    out = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 1, 2), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)
    out = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    agg = np.zeros((4, 5), dtype='i4')
    draw_triangle(tri, (1, 3, 1, 1), (0, 0, 0), (agg,), 1)
    np.testing.assert_equal(agg, out)