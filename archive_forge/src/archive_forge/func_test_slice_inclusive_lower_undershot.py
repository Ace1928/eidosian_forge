from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Histogram, QuadMesh
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_slice_inclusive_lower_undershot(self):
    """Inclusive lower boundary semantics for bin centers"""
    sliced = self.dataset1d[3.45:]
    self.assertEqual(sliced.dimension_values(1), np.arange(3, 10))
    self.assertEqual(sliced.edges, np.arange(3, 11))