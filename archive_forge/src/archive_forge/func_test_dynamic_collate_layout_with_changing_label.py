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
def test_dynamic_collate_layout_with_changing_label(self):

    def callback(i):
        return Layout([Curve([], label=str(j)) for j in range(i, i + 2)])
    dmap = DynamicMap(callback, kdims=['i']).redim.range(i=(0, 10))
    layout = dmap.collate()
    dmap1, dmap2 = layout.values()
    el1, el2 = (dmap1[2], dmap2[2])
    self.assertEqual(el1.label, '2')
    self.assertEqual(el2.label, '3')