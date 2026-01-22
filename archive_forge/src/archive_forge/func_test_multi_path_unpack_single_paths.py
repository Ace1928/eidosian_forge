import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_multi_path_unpack_single_paths(self):
    path = Path([Path([(0, 1), (1, 2)]), Path([(2, 3), (3, 4)])])
    self.assertTrue(path.interface.multi)
    self.assertEqual(path.dimension_values(0), np.array([0, 1, np.nan, 2, 3]))
    self.assertEqual(path.dimension_values(1), np.array([1, 2, np.nan, 3, 4]))