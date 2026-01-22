import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
def test_depth_mismatch(self):
    try:
        self.assertEqual(self.overlay1_depth2, self.overlay4_depth3)
    except AssertionError as e:
        self.assertEqual(str(e), 'Overlays have mismatched path counts.')