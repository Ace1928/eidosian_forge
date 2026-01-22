from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_points_selection_categorical(self):
    points = Points((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
    expr, bbox, region = points._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'], y_selection=None)
    self.assertEqual(bbox, {'x': ['B', 'A', 'C'], 'y': (1, 3)})
    self.assertEqual(expr.apply(points), np.array([True, True, True, False, False]))
    self.assertEqual(region, Rectangles([(0, 1, 2, 3)]) * Path([]))