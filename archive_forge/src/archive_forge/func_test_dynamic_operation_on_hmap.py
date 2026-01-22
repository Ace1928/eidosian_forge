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
def test_dynamic_operation_on_hmap(self):
    hmap = HoloMap({i: Image(sine_array(0, i)) for i in range(10)})
    dmap = Dynamic(hmap, operation=lambda x: x)
    self.assertEqual(dmap.kdims[0].name, hmap.kdims[0].name)
    self.assertEqual(dmap.kdims[0].values, hmap.keys())