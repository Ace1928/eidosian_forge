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
def test_clashing_rename_constructor(self):
    regexp = '(.+?)parameter of the same name'
    with self.assertRaisesRegex(KeyError, regexp):
        PointerXY(rename={'x': 'xtest', 'y': 'x'}, x=0, y=4)