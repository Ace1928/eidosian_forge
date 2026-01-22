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
def test_param_stream_instance(self):
    inner = self.inner(x=2)
    stream = Params(inner)
    self.assertEqual(set(stream.parameters), {inner.param.x, inner.param.y})
    self.assertEqual(stream.contents, {'x': 2, 'y': 0})
    values = []

    def subscriber(**kwargs):
        values.append(kwargs)
    stream.add_subscriber(subscriber)
    inner.y = 2
    self.assertEqual(values, [{'x': 2, 'y': 2}])