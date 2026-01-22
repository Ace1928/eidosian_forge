import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_operation_grid(self):
    grid = GridSpace({i: Image(np.random.rand(10, 10)) for i in range(10)}, kdims=['X'])
    op_grid = operation(grid, op=lambda x, k: x.clone(x.data * 2))
    doubled = grid.clone({k: v.clone(v.data * 2, group='Operation') for k, v in grid.items()})
    self.assertEqual(op_grid, doubled)