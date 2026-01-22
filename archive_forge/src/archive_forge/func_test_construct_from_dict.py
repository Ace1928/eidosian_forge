from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_construct_from_dict(self):
    dataset = Dataset((self.xs, self.ys, self.zs), ['x', 'y'], 'z')
    self.assertEqual(dataset.dimension_values('x'), self.xs.T.flatten())
    self.assertEqual(dataset.dimension_values('y'), self.ys.T.flatten())
    self.assertEqual(dataset.dimension_values('z'), self.zs.T.flatten())