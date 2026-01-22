import numpy as np
from holoviews.element.comparison import ComparisonTestCase
from shapely.geometry import (
from geoviews.element import Rectangles, Path, Polygons, Points, Segments
def test_single_geom_conversion(self):
    segs = Segments([(0, 0, 1, 1)])
    geom = segs.geom()
    self.assertIsInstance(geom, LineString)
    self.assertEqual(np.array(geom.coords), np.array([[0, 0], [1, 1]]))