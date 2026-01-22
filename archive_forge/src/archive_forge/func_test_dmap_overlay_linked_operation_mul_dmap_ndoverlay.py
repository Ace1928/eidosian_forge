import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_dmap_overlay_linked_operation_mul_dmap_ndoverlay(self):
    mapped = operation(self.dmap_overlay, link_inputs=True)
    test = mapped * self.dmap_ndoverlay
    initialize_dynamic(test)
    layers = [mapped, mapped, self.dmap_ndoverlay]
    self.assertEqual(split_dmap_overlay(test), layers)