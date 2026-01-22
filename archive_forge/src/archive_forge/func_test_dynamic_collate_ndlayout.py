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
def test_dynamic_collate_ndlayout(self):

    def callback():
        return NdLayout({i: Image(np.array([[i, 1], [2, 3]])) for i in range(1, 3)})
    dmap = DynamicMap(callback, kdims=[])
    layout = dmap.collate()
    self.assertEqual(list(layout.keys()), [1, 2])
    self.assertEqual(layout[1][()], Image(np.array([[1, 1], [2, 3]])))