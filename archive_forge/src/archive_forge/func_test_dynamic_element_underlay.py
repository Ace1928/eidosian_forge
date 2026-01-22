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
def test_dynamic_element_underlay(self):
    fn = lambda i: Image(sine_array(0, i))
    dmap = DynamicMap(fn, kdims=['i'])
    dynamic_overlay = Image(sine_array(0, 10)) * dmap
    overlaid = Image(sine_array(0, 10)) * Image(sine_array(0, 5))
    self.assertEqual(dynamic_overlay[5], overlaid)