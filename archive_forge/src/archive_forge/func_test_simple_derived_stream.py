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
def test_simple_derived_stream(self):
    v0 = Val(v=1.0)
    v1 = Val(v=2.0)
    s0 = Sum([v0, v1])
    self.assertEqual(s0.v, 3.0)
    v0.event(v=7.0)
    self.assertEqual(s0.v, 9.0)
    v1.event(v=-8.0)
    self.assertEqual(s0.v, -1.0)