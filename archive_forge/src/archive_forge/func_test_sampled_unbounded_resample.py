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
def test_sampled_unbounded_resample(self):
    fn = lambda i: Image(sine_array(0, i))
    dmap = DynamicMap(fn, kdims=['i'])
    self.assertEqual(dmap[{0, 1, 2}].keys(), [0, 1, 2])
    self.assertEqual(dmap.unbounded, ['i'])