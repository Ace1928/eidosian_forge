from unittest import mock
from glance.async_ import utils
import glance.common.exception
from glance.tests.unit import base
def test_return_matching_glance_endpoint(self):
    self.assertEqual(utils.get_glance_endpoint(self.context, 'RegionOne', 'public'), 'http://RegionOnePublic/')
    self.assertEqual(utils.get_glance_endpoint(self.context, 'RegionTwo', 'internal'), 'http://RegionTwoInternal/')