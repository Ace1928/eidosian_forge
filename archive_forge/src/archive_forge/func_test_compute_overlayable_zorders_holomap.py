import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_compute_overlayable_zorders_holomap(self):
    hmap = HoloMap({0: Points([])})
    sources = compute_overlayable_zorders(hmap)
    self.assertEqual(sources[0], [hmap, hmap.last])