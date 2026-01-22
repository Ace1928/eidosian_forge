import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_groupby_unsorted(self):
    data = [(('B', 2), 1), (('C', 2), 2), (('A', 1), 3)]
    grouped = NdMapping(data, sort=False, kdims=['X', 'Y']).groupby('Y')
    self.assertEqual(grouped.keys(), [2, 1])
    self.assertEqual(grouped.values()[0].keys(), ['B', 'C'])
    self.assertEqual(grouped.last.keys(), ['A'])