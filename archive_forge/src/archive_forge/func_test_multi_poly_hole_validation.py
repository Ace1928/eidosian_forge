import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_multi_poly_hole_validation(self):
    xs = [1, 2, 3, np.nan, 6, 7, 3]
    ys = [2, 0, 7, np.nan, 7, 5, 2]
    with self.assertRaises(DataError):
        Polygons([{'x': xs, 'y': ys, 'holes': [[]]}])