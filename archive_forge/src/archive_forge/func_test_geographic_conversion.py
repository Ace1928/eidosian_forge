import unittest
from holoviews.core import HoloMap
from holoviews.element import Curve
from geoviews.element import is_geographic, Image, Dataset
from geoviews.element.comparison import ComparisonTestCase
def test_geographic_conversion(self):
    self.assertEqual(Dataset(self.cube, kdims=['longitude', 'latitude']).to.image(), Image(self.cube, kdims=['longitude', 'latitude']))