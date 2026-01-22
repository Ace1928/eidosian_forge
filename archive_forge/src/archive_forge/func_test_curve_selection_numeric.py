from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_curve_selection_numeric(self):
    curve = Curve([3, 2, 1, 3, 4])
    expr, bbox, region = curve._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2))
    self.assertEqual(bbox, {'x': (1, 3)})
    self.assertEqual(expr.apply(curve), np.array([False, True, True, True, False]))
    self.assertEqual(region, NdOverlay({0: VSpan(1, 3)}))