from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_poly_selection_numeric(self):
    poly = Polygons([[(0, 0), (0.2, 0.1), (0.3, 0.4), (0.1, 0.2)], [(0.25, -0.1), (0.4, 0.2), (0.6, 0.3), (0.5, 0.1)], [(0.3, 0.3), (0.5, 0.4), (0.6, 0.5), (0.35, 0.45)]])
    expr, bbox, region = poly._get_selection_expr_for_stream_value(bounds=(0.2, -0.2, 0.6, 0.6))
    self.assertEqual(bbox, {'x': (0.2, 0.6), 'y': (-0.2, 0.6)})
    self.assertEqual(expr.apply(poly, expanded=False), np.array([False, True, True]))
    self.assertEqual(region, Rectangles([(0.2, -0.2, 0.6, 0.6)]) * Path([]))