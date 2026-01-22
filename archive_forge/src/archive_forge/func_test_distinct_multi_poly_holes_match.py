import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_distinct_multi_poly_holes_match(self):
    self.assertTrue(self.distinct_polys.interface.has_holes(self.distinct_polys))
    paths = self.distinct_polys.split(datatype='array')
    holes = self.distinct_polys.interface.holes(self.distinct_polys)
    self.assertEqual(len(paths), len(holes))
    self.assertEqual(len(holes), 2)
    self.assertEqual(len(holes[0]), 2)
    self.assertEqual(len(holes[0][0]), 2)
    self.assertEqual(len(holes[0][1]), 0)
    self.assertEqual(len(holes[1]), 1)
    self.assertEqual(len(holes[1][0]), 0)