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
def test_custom_types(self):
    self.assertEqual(isinstance(self.TypesTest.param['t'], param.Boolean), True)
    self.assertEqual(isinstance(self.TypesTest.param['u'], param.Integer), True)
    self.assertEqual(isinstance(self.TypesTest.param['v'], param.Number), True)
    self.assertEqual(isinstance(self.TypesTest.param['w'], param.Tuple), True)
    self.assertEqual(isinstance(self.TypesTest.param['x'], param.String), True)
    self.assertEqual(isinstance(self.TypesTest.param['y'], param.List), True)
    self.assertEqual(isinstance(self.TypesTest.param['z'], param.Array), True)