from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests
def test_multi_polygon_constructor(self):
    xs = [1, 2, 3, np.nan, 6, 7, 3]
    ys = [2, 0, 7, np.nan, 7, 5, 2]
    holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
    path = Polygons([{'x': xs, 'y': ys, 'holes': holes}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
    self.assertIsInstance(path.data.geometry.dtype, MultiPolygonDtype)
    self.assertEqual(path.data.iloc[0, 0].buffer_values, np.array([1.0, 2.0, 2.0, 0.0, 3.0, 7.0, 1.0, 2.0, 1.5, 2.0, 2.0, 3.0, 1.6, 1.6, 1.5, 2.0, 2.1, 4.5, 2.5, 5.0, 2.3, 3.5, 2.1, 4.5, 6.0, 7.0, 3.0, 2.0, 7.0, 5.0, 6.0, 7.0]))
    self.assertEqual(path.data.iloc[1, 0].buffer_values, np.array([3, 2, 7, 5, 6, 7, 3, 2, 3, 7, 1, 2, 2, 0, 3, 7]))