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
def test_selection_expr_stream_1D_elements(self):
    element_type_1D = [Scatter]
    for element_type in element_type_1D:
        element = element_type(([1, 2, 3], [1, 5, 10]))
        expr_stream = SelectionExpr(element)
        self.assertEqual(len(expr_stream.input_streams), 1)
        self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
        self.assertIsNone(expr_stream.bbox)
        self.assertIsNone(expr_stream.selection_expr)
        expr_stream.input_streams[0].event(bounds=(1, 1, 3, 4))
        self.assertEqual(repr(expr_stream.selection_expr), repr((dim('x') >= 1) & (dim('x') <= 3)))
        self.assertEqual(expr_stream.bbox, {'x': (1, 3)})