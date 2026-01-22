import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_ndmapping_slice_upper_bound_exclusive2_float(self):
    ndmap = NdMapping(self.init_item_odict, kdims=[self.dim1, self.dim2])
    self.assertEqual(ndmap[:, 0.0:3.0].keys(), [(1, 2.0)])