import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_multi_poly_no_holes_match(self):
    self.assertFalse(self.multi_poly_no_hole.interface.has_holes(self.multi_poly_no_hole))
    paths = self.multi_poly_no_hole.split(datatype='array')
    holes = self.multi_poly_no_hole.interface.holes(self.multi_poly_no_hole)
    self.assertEqual(len(paths), len(holes))
    self.assertEqual(len(holes), 1)
    self.assertEqual(len(holes[0]), 2)
    self.assertEqual(len(holes[0][0]), 0)
    self.assertEqual(len(holes[0][1]), 0)