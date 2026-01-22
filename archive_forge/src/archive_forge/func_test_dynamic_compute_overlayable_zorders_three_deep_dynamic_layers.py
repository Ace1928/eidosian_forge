import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_dynamic_compute_overlayable_zorders_three_deep_dynamic_layers(self):
    area = DynamicMap(lambda: Area(range(10)), kdims=[])
    curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
    curve2 = DynamicMap(lambda: Curve(range(10)), kdims=[])
    area_redim = area.redim(x='x2')
    curve_redim = curve.redim(x='x2')
    curve2_redim = curve2.redim(x='x3')
    combined = area_redim * curve_redim
    combined1 = combined * curve2_redim
    combined1[()]
    sources = compute_overlayable_zorders(combined1)
    self.assertIn(area_redim, sources[0])
    self.assertIn(area, sources[0])
    self.assertNotIn(curve_redim, sources[0])
    self.assertNotIn(curve, sources[0])
    self.assertNotIn(curve2_redim, sources[0])
    self.assertNotIn(curve2, sources[0])
    self.assertIn(curve_redim, sources[1])
    self.assertIn(curve, sources[1])
    self.assertNotIn(area_redim, sources[1])
    self.assertNotIn(area, sources[1])
    self.assertNotIn(curve2_redim, sources[1])
    self.assertNotIn(curve2, sources[1])
    self.assertIn(curve2_redim, sources[2])
    self.assertIn(curve2, sources[2])
    self.assertNotIn(area_redim, sources[2])
    self.assertNotIn(area, sources[2])
    self.assertNotIn(curve_redim, sources[2])
    self.assertNotIn(curve, sources[2])