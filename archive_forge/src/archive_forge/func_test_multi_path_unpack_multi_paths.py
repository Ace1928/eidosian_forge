import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_multi_path_unpack_multi_paths(self):
    path = Path([Path([[(0, 1), (1, 2)]]), Path([[(2, 3), (3, 4)], [(4, 5), (5, 6)]])])
    self.assertTrue(path.interface.multi)
    self.assertEqual(path.dimension_values(0), np.array([0, 1, np.nan, 2, 3, np.nan, 4, 5]))
    self.assertEqual(path.dimension_values(1), np.array([1, 2, np.nan, 3, 4, np.nan, 5, 6]))