from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_groupby_ydim(self):
    grouped = self.dataset2d.groupby('y', group_type=Dataset)
    holomap = HoloMap({self.ys[i:i + 2].mean(): Dataset((self.xs, self.zs[i]), 'x', 'z') for i in range(3)}, kdims=['y'])
    self.assertEqual(grouped, holomap)