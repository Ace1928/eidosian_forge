from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_set_base_image_properties_none(self):
    properties = None
    script_utils.set_base_image_properties(properties)
    self.assertIsNone(properties)