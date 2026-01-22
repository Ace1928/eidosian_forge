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
def test_selection_expr_stream_polygon_index_cols(self):
    try:
        import shapely
    except ImportError:
        try:
            import spatialpandas
        except ImportError:
            raise SkipTest('Shapely required for polygon selection')
    poly = Polygons([[(0, 0, 'a'), (2, 0, 'a'), (1, 1, 'a')], [(2, 0, 'b'), (4, 0, 'b'), (3, 1, 'b')], [(1, 1, 'c'), (3, 1, 'c'), (2, 2, 'c')]], vdims=['cat'])
    events = []
    expr_stream = SelectionExpr(poly, index_cols=['cat'])
    expr_stream.add_subscriber(lambda **kwargs: events.append(kwargs))
    self.assertEqual(len(expr_stream.input_streams), 3)
    self.assertIsInstance(expr_stream.input_streams[0], SelectionXY)
    self.assertIsInstance(expr_stream.input_streams[1], Lasso)
    self.assertIsInstance(expr_stream.input_streams[2], Selection1D)
    self.assertIsNone(expr_stream.bbox)
    self.assertIsNone(expr_stream.selection_expr)
    expr_stream.input_streams[2].event(index=[0, 1])
    self.assertEqual(repr(expr_stream.selection_expr), repr(dim('cat').isin(['a', 'b'])))
    self.assertEqual(expr_stream.bbox, None)
    self.assertEqual(len(events), 1)
    expr_stream.input_streams[0].event(bounds=(0, 0, 4, 1))
    self.assertEqual(repr(expr_stream.selection_expr), repr(dim('cat').isin(['a', 'b'])))
    self.assertEqual(len(events), 1)
    expr_stream.input_streams[1].event(geometry=np.array([(0, 0), (4, 0), (4, 2), (0, 2)]))
    self.assertEqual(repr(expr_stream.selection_expr), repr(dim('cat').isin(['a', 'b', 'c'])))
    self.assertEqual(len(events), 2)
    expr_stream.input_streams[2].event(index=[1, 2])
    self.assertEqual(repr(expr_stream.selection_expr), repr(dim('cat').isin(['b', 'c'])))
    self.assertEqual(expr_stream.bbox, None)
    self.assertEqual(len(events), 3)