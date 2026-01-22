from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_irregular_transform_replace_vdim(self):
    transformed = Dataset((self.xs, self.ys, self.zs), ['x', 'y'], 'z').transform(z=dim('z') * 2)
    expected = Dataset((self.xs, self.ys, self.zs * 2), ['x', 'y'], 'z')
    self.assertEqual(expected, transformed)