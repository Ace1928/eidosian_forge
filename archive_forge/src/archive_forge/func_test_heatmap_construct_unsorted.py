import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_heatmap_construct_unsorted(self):
    hmap = HeatMap([('B', 'b', 2), ('A', 'a', 1)])
    dataset = Dataset({'x': ['B', 'A'], 'y': ['b', 'a'], 'z': [[2, np.nan], [np.nan, 1]]}, kdims=['x', 'y'], vdims=['z'], label='unique')
    self.assertEqual(hmap.gridded, dataset)