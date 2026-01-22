from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews.core import NdOverlay
from holoviews.core.options import Store
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.selection import spatial_select_columnar
def test_segs_selection_numeric_inverted(self):
    segs = Segments([(0, 1, 2, 3), (1, 3, 1.5, 4), (2.5, 4.2, 3.5, 4.8)]).opts(invert_axes=True)
    expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.9, 0.5, 4.9, 3.4))
    self.assertEqual(bbox, {'x0': (0.5, 3.4), 'y0': (0.9, 4.9), 'x1': (0.5, 3.4), 'y1': (0.9, 4.9)})
    self.assertEqual(expr.apply(segs), np.array([False, True, False]))
    self.assertEqual(region, Rectangles([(0.9, 0.5, 4.9, 3.4)]) * Path([]))
    expr, bbox, region = segs._get_selection_expr_for_stream_value(bounds=(0.9, 0, 4.9, 3.5))
    self.assertEqual(bbox, {'x0': (0, 3.5), 'y0': (0.9, 4.9), 'x1': (0, 3.5), 'y1': (0.9, 4.9)})
    self.assertEqual(expr.apply(segs), np.array([True, True, True]))
    self.assertEqual(region, Rectangles([(0.9, 0, 4.9, 3.5)]) * Path([]))