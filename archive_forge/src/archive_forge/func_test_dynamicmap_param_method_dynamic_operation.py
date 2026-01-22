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
def test_dynamicmap_param_method_dynamic_operation(self):
    inner = self.inner()
    dmap = DynamicMap(inner.method)
    inner_stream = dmap.streams[0]
    op_dmap = Dynamic(dmap, operation=inner.op_method)
    self.assertEqual(len(op_dmap.streams), 1)
    stream = op_dmap.streams[0]
    self.assertEqual(set(stream.parameters), {inner.param.y})
    self.assertIsInstance(stream, ParamMethod)
    self.assertEqual(stream.contents, {})
    values_x, values_y = ([], [])

    def subscriber_x(**kwargs):
        values_x.append(kwargs)

    def subscriber_y(**kwargs):
        values_y.append(kwargs)
    inner_stream.add_subscriber(subscriber_x)
    stream.add_subscriber(subscriber_y)
    inner.y = 3
    self.assertEqual(values_x, [])
    self.assertEqual(values_y, [{}])