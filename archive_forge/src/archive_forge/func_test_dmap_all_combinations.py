import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_dmap_all_combinations(self):
    test = self.dmap_overlay * self.element * self.dmap_ndoverlay * self.overlay * self.dmap_element * self.ndoverlay
    initialize_dynamic(test)
    layers = [self.dmap_overlay, self.dmap_overlay, self.element, self.dmap_ndoverlay, self.el1, self.el2, self.dmap_element, self.ndoverlay]
    self.assertEqual(split_dmap_overlay(test), layers)