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
def test_point_bounds_check():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [5, 6, 7]})
    p = Point('x', 'y')
    assert p._compute_bounds(df['x'].values) == (1, 3)
    assert p._compute_bounds(df['y'].values) == (5, 7)