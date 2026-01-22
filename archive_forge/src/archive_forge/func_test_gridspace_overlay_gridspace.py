import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_gridspace_overlay_gridspace(self):
    items = [(0, self.view1), (1, self.view2), (2, self.view3), (3, self.view2)]
    grid = GridSpace(items, 'X')
    items2 = [(0, self.view2), (1, self.view1), (2, self.view2), (3, self.view3)]
    grid2 = GridSpace(items2, 'X')
    expected_items = [(0, self.view1 * self.view2), (1, self.view2 * self.view1), (2, self.view3 * self.view2), (3, self.view2 * self.view3)]
    expected = GridSpace(expected_items, 'X')
    self.assertEqual(grid * grid2, expected)