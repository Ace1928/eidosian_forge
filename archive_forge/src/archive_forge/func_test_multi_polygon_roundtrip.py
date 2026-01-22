from unittest import SkipTest
import numpy as np
import pandas as pd
from shapely import geometry as sgeom
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path, Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import GeomTests
from geoviews.data import GeoPandasInterface
from .test_multigeometry import GeomInterfaceTest
def test_multi_polygon_roundtrip(self):
    xs = [1, 2, 3, np.nan, 6, 7, 3]
    ys = [2, 0, 7, np.nan, 7, 5, 2]
    holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
    poly = Polygons([{'x': xs, 'y': ys, 'holes': holes, 'z': 1}, {'x': xs[::-1], 'y': ys[::-1], 'z': 2}], ['x', 'y'], 'z', datatype=[self.datatype])
    self.assertIsInstance(poly.data.geometry.dtype, GeometryDtype)
    roundtrip = poly.clone(datatype=['multitabular'])
    self.assertEqual(roundtrip.interface.datatype, 'multitabular')
    expected = Polygons([{'x': [1, 2, 3, 1, np.nan, 6, 7, 3, 6], 'y': [2, 0, 7, 2, np.nan, 7, 5, 2, 7], 'holes': holes, 'z': 1}, {'x': [3, 7, 6, 3, np.nan, 3, 2, 1, 3], 'y': [2, 5, 7, 2, np.nan, 7, 0, 2, 7], 'z': 2}], ['x', 'y'], 'z', datatype=['multitabular'])
    self.assertEqual(roundtrip, expected)