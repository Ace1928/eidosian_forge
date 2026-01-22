import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_multi_poly_holes_match(self):
    self.assertTrue(self.multi_poly.interface.has_holes(self.multi_poly))
    paths = self.multi_poly.split(datatype='array')
    holes = self.multi_poly.interface.holes(self.multi_poly)
    self.assertEqual(len(paths), len(holes))
    self.assertEqual(len(holes), 1)
    self.assertEqual(len(holes[0]), 2)
    self.assertEqual(len(holes[0][0]), 2)
    self.assertEqual(len(holes[0][1]), 0)