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
def test_dynamic_collate_layout(self):

    def callback():
        return Image(np.array([[0, 1], [2, 3]])) + Text(0, 0, 'Test')
    dmap = DynamicMap(callback, kdims=[])
    layout = dmap.collate()
    self.assertEqual(list(layout.keys()), [('Image', 'I'), ('Text', 'I')])
    self.assertEqual(layout.Image.I[()], Image(np.array([[0, 1], [2, 3]])))