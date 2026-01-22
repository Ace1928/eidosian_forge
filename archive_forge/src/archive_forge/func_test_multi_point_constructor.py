from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests
def test_multi_point_constructor(self):
    xs = [1, 2, 3, 2]
    ys = [2, 0, 7, 4]
    points = Points([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
    self.assertIsInstance(points.data.geometry.dtype, MultiPointDtype)
    self.assertEqual(points.data.iloc[0, 0].buffer_values, np.array([1, 2, 2, 0, 3, 7, 2, 4]))
    self.assertEqual(points.data.iloc[1, 0].buffer_values, np.array([2, 4, 3, 7, 2, 0, 1, 2]))