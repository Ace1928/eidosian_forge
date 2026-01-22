import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_grid_index_one_axis(self):
    vals = [self.view1, self.view2, self.view3, self.view2]
    keys = [('A', 0), ('B', 1), ('C', 0), ('D', 1)]
    grid = GridSpace(zip(keys, vals))
    self.assertEqual(grid[:, 0], GridSpace([(('A', 0), self.view1), (('C', 0), self.view3)]))