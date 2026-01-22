from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_scatter_selection_categorical(self):
    scatter = Scatter((['B', 'A', 'C', 'D', 'E'], [3, 2, 1, 3, 4]))
    expr, bbox, region = scatter._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 3), x_selection=['B', 'A', 'C'], y_selection=None)
    self.assertEqual(bbox, {'x': ['B', 'A', 'C']})
    self.assertEqual(expr.apply(scatter), np.array([True, True, True, False, False]))
    self.assertEqual(region, NdOverlay({0: VSpan(0, 2)}))