import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_unsorted(self):
    data = [('B', 1), ('C', 2), ('A', 3)]
    ndmap = MultiDimensionalMapping(data, sort=False)
    self.assertEqual(ndmap.keys(), ['B', 'C', 'A'])