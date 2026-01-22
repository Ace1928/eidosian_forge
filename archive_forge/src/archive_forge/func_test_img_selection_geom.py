from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
@ds_available
@spd_available
def test_img_selection_geom(self):
    img = Image(([0, 1, 2], [0, 1, 2, 3], np.random.rand(4, 3)))
    geom = np.array([(-0.4, -0.1), (0.6, -0.1), (0.4, 1.7), (-0.1, 1.7)])
    expr, bbox, region = img._get_selection_expr_for_stream_value(geometry=geom)
    self.assertEqual(bbox, {'x': np.array([-0.4, 0.6, 0.4, -0.1]), 'y': np.array([-0.1, -0.1, 1.7, 1.7])})
    self.assertEqual(expr.apply(img, expanded=True, flat=False), np.array([[1.0, np.nan, np.nan], [1.0, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]))
    self.assertEqual(region, Rectangles([]) * Path([list(geom) + [(-0.4, -0.1)]]))