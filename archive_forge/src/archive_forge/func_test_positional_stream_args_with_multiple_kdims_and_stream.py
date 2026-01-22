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
def test_positional_stream_args_with_multiple_kdims_and_stream(self):
    fn = lambda i, j, s1, s2: Points([i, j]) + Curve([s1['x'], s2['y']])
    x_stream = X(x=2)
    y_stream = Y(y=3)
    dmap = DynamicMap(fn, kdims=['i', 'j'], streams=[x_stream, y_stream], positional_stream_args=True)
    self.assertEqual(dmap[0, 1], Points([0, 1]) + Curve([2, 3]))
    x_stream.event(x=5)
    y_stream.event(y=6)
    self.assertEqual(dmap[3, 4], Points([3, 4]) + Curve([5, 6]))