from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_area_selection_numeric_index_cols(self):
    area = Area([3, 2, 1, 3, 2])
    expr, bbox, region = area._get_selection_expr_for_stream_value(bounds=(1, 0, 3, 2), index_cols=['y'])
    self.assertEqual(bbox, {'x': (1, 3)})
    self.assertEqual(expr.apply(area), np.array([False, True, True, False, True]))
    self.assertEqual(region, None)