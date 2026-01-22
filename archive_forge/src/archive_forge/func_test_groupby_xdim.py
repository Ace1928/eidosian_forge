from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_groupby_xdim(self):
    grouped = self.dataset2d.groupby('x', group_type=Dataset)
    holomap = HoloMap({(self.xs[i] + np.diff(self.xs[i:i + 2]) / 2.0)[0]: Dataset((self.ys, self.zs[:, i]), 'y', 'z') for i in range(3)}, kdims=['x'])
    self.assertEqual(grouped, holomap)