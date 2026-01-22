import unittest
from holoviews.core import HoloMap
from holoviews.element import Curve
from geoviews.element import is_geographic, Image, Dataset
from geoviews.element.comparison import ComparisonTestCase
def test_is_geographic_2d(self):
    self.assertTrue(is_geographic(Dataset(self.cube, kdims=['longitude', 'latitude']), ['longitude', 'latitude']))