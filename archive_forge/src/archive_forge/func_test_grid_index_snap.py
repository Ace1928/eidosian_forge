import numpy as np
import pytest
from holoviews import (
from holoviews.element import Curve, HLine, Image
from holoviews.element.comparison import ComparisonTestCase
def test_grid_index_snap(self):
    vals = [self.view1, self.view2, self.view3, self.view2]
    keys = [(0, 0), (0, 1), (1, 0), (1, 1)]
    grid = GridSpace(zip(keys, vals))
    self.assertEqual(grid[0.1, 0.1], self.view1)