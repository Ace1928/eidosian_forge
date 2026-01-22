import time
import uuid
from collections import deque
import numpy as np
import param
import pytest
from holoviews import Dimension, GridSpace, Layout, NdLayout, NdOverlay
from holoviews.core.options import Store
from holoviews.core.spaces import Callable, DynamicMap, HoloMap
from holoviews.element import Curve, Image, Points, Scatter, Text
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import histogram
from holoviews.plotting.util import initialize_dynamic
from holoviews.streams import (
from holoviews.util import Dynamic
from ..utils import LoggingComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_deep_map_transform_element_type(self):
    fn = lambda i: Curve(np.arange(i))
    dmap = DynamicMap(fn, kdims=[Dimension('Test', range=(10, 20))])
    dmap[10]
    mapped = dmap.map(lambda x: Scatter(x), Curve)
    area = mapped[11]
    self.assertEqual(area, Scatter(fn(11)))