import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_multi_path_list_split(self):
    path = Path([[(0, 1), (1, 2)], [(2, 3), (3, 4)]])
    subpaths = path.split()
    self.assertEqual(len(subpaths), 2)
    self.assertEqual(subpaths[0], Path([(0, 1), (1, 2)]))
    self.assertEqual(subpaths[1], Path([(2, 3), (3, 4)]))