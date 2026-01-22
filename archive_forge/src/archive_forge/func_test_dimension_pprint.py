import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_pprint(self):
    dim = Dimension('Test dimension', cyclic=True, type=float, unit='Twilight zones')
    self.assertEqual(dim.pprint_value_string(3.23451), 'Test dimension: 3.2345 Twilight zones')
    self.assertEqual(dim.pprint_value_string(4.23441), 'Test dimension: 4.2344 Twilight zones')
    self.assertEqual(dim.pprint_value(3.23451, print_unit=True), '3.2345 Twilight zones')
    self.assertEqual(dim.pprint_value(4.23441, print_unit=True), '4.2344 Twilight zones')