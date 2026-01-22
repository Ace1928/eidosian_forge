from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_violin_cats(self):
    violin = Violin((['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C', 'C'], list(range(10))), 'x', 'y')
    expr, bbox, region = violin._get_selection_expr_for_stream_value(bounds=(0, 1, 2, 7), x_selection=['A', 'B'])
    self.assertEqual(bbox, {'y': (1, 7), 'x': ['A', 'B']})
    self.assertEqual(expr.apply(violin), np.array([False, True, True, True, True, False, False, False, False, False]))
    self.assertEqual(region, NdOverlay({0: HSpan(1, 7)}))