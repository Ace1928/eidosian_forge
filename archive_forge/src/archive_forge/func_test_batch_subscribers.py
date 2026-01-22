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
def test_batch_subscribers(self):
    subscriber1 = _TestSubscriber()
    subscriber2 = _TestSubscriber()
    positionX = PointerX(subscribers=[subscriber1, subscriber2])
    positionY = PointerY(subscribers=[subscriber1, subscriber2])
    positionX.update(x=50)
    positionY.update(y=100)
    Stream.trigger([positionX, positionY])
    self.assertEqual(subscriber1.kwargs, dict(x=50, y=100))
    self.assertEqual(subscriber1.call_count, 1)
    self.assertEqual(subscriber2.kwargs, dict(x=50, y=100))
    self.assertEqual(subscriber2.call_count, 1)