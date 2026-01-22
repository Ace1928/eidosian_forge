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
def test_dynamicmap_param_method_deps_memoization(self):
    inner = self.inner()
    dmap = DynamicMap(inner.method)
    stream = dmap.streams[0]
    self.assertEqual(set(stream.parameters), {inner.param.x})
    self.assertEqual(stream.contents, {})
    dmap[()]
    dmap[()]
    self.assertEqual(inner.count, 1)