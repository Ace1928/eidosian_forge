from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests
def test_point_constructor(self):
    points = Points([{'x': 0, 'y': 1}, {'x': 1, 'y': 0}], ['x', 'y'], datatype=[self.datatype])
    self.assertIsInstance(points.data.geometry.dtype, PointDtype)
    self.assertEqual(points.data.iloc[0, 0].flat_values, np.array([0, 1]))
    self.assertEqual(points.data.iloc[1, 0].flat_values, np.array([1, 0]))