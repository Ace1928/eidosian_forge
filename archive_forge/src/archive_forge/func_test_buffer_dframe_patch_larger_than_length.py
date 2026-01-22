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
def test_buffer_dframe_patch_larger_than_length(self):
    data = pd.DataFrame({'x': np.array([0]), 'y': np.array([1])})
    buff = Buffer(data, length=1, index=False)
    chunk = pd.DataFrame({'x': np.array([1, 2]), 'y': np.array([2, 3])})
    buff.send(chunk)
    dframe = pd.DataFrame({'x': np.array([2]), 'y': np.array([3])})
    self.assertEqual(buff.data.values, dframe.values)