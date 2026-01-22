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
def test_subscribers(self):
    subscriber1 = _TestSubscriber()
    subscriber2 = _TestSubscriber()
    position = PointerXY(subscribers=[subscriber1, subscriber2])
    kwargs = dict(x=3, y=4)
    position.event(**kwargs)
    self.assertEqual(subscriber1.kwargs, kwargs)
    self.assertEqual(subscriber2.kwargs, kwargs)