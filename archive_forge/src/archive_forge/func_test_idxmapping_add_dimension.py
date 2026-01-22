import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_add_dimension(self):
    ndmap = MultiDimensionalMapping(self.init_items_1D_list, kdims=[self.dim1])
    ndmap2d = ndmap.add_dimension(self.dim2, 0, 0.5)
    self.assertEqual(list(ndmap2d.keys()), [(0.5, 1), (0.5, 5)])
    self.assertEqual(ndmap2d.kdims, [self.dim2, self.dim1])