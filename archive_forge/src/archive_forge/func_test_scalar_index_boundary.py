from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_scalar_index_boundary(self):
    """
        Scalar at boundary indexes next bin.
        (exclusive upper boundary for current bin)
        """
    self.assertEqual(self.dataset1d[4], 4)
    self.assertEqual(self.dataset1d[5], 5)