import itertools
import numpy as np
from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_hmap_collate(self):
    hmap = HoloMap({i: Curve(np.arange(10) * i) for i in range(3)})
    overlaid = Overlay([hmap, hmap, hmap]).collate()
    self.assertEqual(overlaid, hmap * hmap * hmap)