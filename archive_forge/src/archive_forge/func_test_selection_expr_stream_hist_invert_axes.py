from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
def test_selection_expr_stream_hist_invert_axes(self):
    hist = Histogram(([1, 2, 3, 4, 5], [1, 5, 2, 3, 7])).opts(invert_axes=True)
    expr_stream = SelectionExpr(hist)
    self.assertEqual(len(expr_stream.input_streams), 1)
    self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
    self.assertIsNone(expr_stream.bbox)
    self.assertIsNone(expr_stream.selection_expr)
    expr_stream.input_streams[0].event(bounds=(2.5, 1.5, 6, 4.6))
    self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1.5) & (dim('x') <= 4.6)))
    self.assertEqual(expr_stream.bbox, {'x': (1.5, 4.6)})
    expr_stream.input_streams[0].event(bounds=(-10, 2.5, 10, 8))
    self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 2.5) & (dim('x') <= 8)))
    self.assertEqual(expr_stream.bbox, {'x': (2.5, 8)})