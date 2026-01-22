import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_dynamic_compute_overlayable_zorders_mixed_dynamic_and_non_dynamic_ndoverlays(self):
    ndoverlay = NdOverlay({i: Area(range(10 + i)) for i in range(2)})
    curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
    curve_redim = curve.redim(x='x2')
    combined = ndoverlay * curve_redim
    combined[()]
    sources = compute_overlayable_zorders(combined)
    self.assertIn(ndoverlay[0], sources[0])
    self.assertIn(ndoverlay, sources[0])
    self.assertNotIn(curve_redim, sources[0])
    self.assertNotIn(curve, sources[0])
    self.assertIn(ndoverlay[1], sources[1])
    self.assertIn(ndoverlay, sources[1])
    self.assertNotIn(curve_redim, sources[1])
    self.assertNotIn(curve, sources[1])
    self.assertIn(curve_redim, sources[2])
    self.assertIn(curve, sources[2])
    self.assertNotIn(ndoverlay, sources[2])