import itertools
import numpy as np
from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_gridspace_collate(self):
    grid = GridSpace({(i, j): Curve(np.arange(10) * i) for i in range(3) for j in range(3)})
    overlaid = Overlay([grid, grid, grid]).collate()
    self.assertEqual(overlaid, grid * grid * grid)