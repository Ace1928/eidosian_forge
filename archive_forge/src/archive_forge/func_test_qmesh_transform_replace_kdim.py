from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_qmesh_transform_replace_kdim(self):
    transformed = self.dataset2d.transform(x=dim('x') * 2)
    expected = QuadMesh((self.xs * 2, self.ys, self.zs))
    self.assertEqual(expected, transformed)