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
def test_lines_xy_validate():
    g = LinesAxis1(['x0', 'x1'], ['y11', 'y12'])
    g.validate(dshape('{x0: int32, x1: int32, y11: float32, y12: float32}'))
    with pytest.raises(ValueError):
        g.validate(dshape('{x0: int32, x1: float32, y11: string, y12: float32}'))