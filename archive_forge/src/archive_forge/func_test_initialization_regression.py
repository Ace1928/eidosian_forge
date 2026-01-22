import time
from tests.unit import unittest
from boto.cloudsearch.layer1 import Layer1
from boto.cloudsearch.layer2 import Layer2
from boto.regioninfo import RegionInfo
def test_initialization_regression(self):
    us_west_2 = RegionInfo(name='us-west-2', endpoint='cloudsearch.us-west-2.amazonaws.com')
    self.layer2 = Layer2(region=us_west_2, host='cloudsearch.us-west-2.amazonaws.com')
    self.assertEqual(self.layer2.layer1.host, 'cloudsearch.us-west-2.amazonaws.com')