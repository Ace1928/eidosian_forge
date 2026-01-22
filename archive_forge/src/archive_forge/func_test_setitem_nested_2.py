import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_setitem_nested_2(self):
    nested1 = MultiDimensionalMapping([('B', 1)])
    ndmap = MultiDimensionalMapping([('A', nested1)])
    nested2 = MultiDimensionalMapping([('C', 2)])
    nested_clone = nested1.clone()
    nested_clone.update(nested2)
    ndmap.update({'A': nested2})
    self.assertEqual(ndmap['A'].data, nested_clone.data)