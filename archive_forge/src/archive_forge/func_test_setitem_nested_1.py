import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_setitem_nested_1(self):
    nested1 = MultiDimensionalMapping([('B', 1)])
    ndmap = MultiDimensionalMapping([('A', nested1)])
    nested2 = MultiDimensionalMapping([('B', 2)])
    ndmap['A'] = nested2
    self.assertEqual(ndmap['A'], nested2)