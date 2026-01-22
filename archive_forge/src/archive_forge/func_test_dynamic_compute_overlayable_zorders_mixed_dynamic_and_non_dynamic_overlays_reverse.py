import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_dynamic_compute_overlayable_zorders_mixed_dynamic_and_non_dynamic_overlays_reverse(self):
    area1 = Area(range(10))
    area2 = Area(range(10))
    overlay = area1 * area2
    curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
    curve_redim = curve.redim(x='x2')
    combined = curve_redim * overlay
    combined[()]
    sources = compute_overlayable_zorders(combined)
    self.assertIn(curve_redim, sources[0])
    self.assertIn(curve, sources[0])
    self.assertNotIn(overlay, sources[0])
    self.assertIn(area1, sources[1])
    self.assertIn(overlay, sources[1])
    self.assertNotIn(curve_redim, sources[1])
    self.assertNotIn(curve, sources[1])
    self.assertIn(area2, sources[2])
    self.assertIn(overlay, sources[2])
    self.assertNotIn(curve_redim, sources[2])
    self.assertNotIn(curve, sources[2])