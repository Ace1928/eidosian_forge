import numpy as np
from holoviews.element.comparison import ComparisonTestCase
from shapely.geometry import (
from geoviews.element import Rectangles, Path, Polygons, Points, Segments
def test_single_geom_with_hole_conversion(self):
    holes = [[((0.5, 0.2), (1, 0.8), (1.5, 0.2))]]
    path = Polygons([{'x': [0, 1, 2], 'y': [0, 1, 0], 'holes': holes}], ['x', 'y'])
    geom = path.geom()
    self.assertIsInstance(geom, Polygon)
    self.assertEqual(np.array(geom.exterior.coords), np.array([[0, 0], [1, 1], [2, 0], [0, 0]]))
    self.assertEqual(len(geom.interiors), 1)
    self.assertIsInstance(geom.interiors[0], LinearRing)
    self.assertEqual(np.array(geom.interiors[0].coords), np.array([[0.5, 0.2], [1, 0.8], [1.5, 0.2], [0.5, 0.2]]))