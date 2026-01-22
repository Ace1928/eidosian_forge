import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_get_axis_padding_tuple_3d(self):
    padding = get_axis_padding((0.1, 0.2, 0.3))
    self.assertEqual(padding, (0.1, 0.2, 0.3))