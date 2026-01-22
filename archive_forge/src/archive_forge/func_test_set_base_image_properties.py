from unittest import mock
import urllib
from glance.common import exception
from glance.common.scripts import utils as script_utils
import glance.tests.utils as test_utils
def test_set_base_image_properties(self):
    properties = {}
    script_utils.set_base_image_properties(properties)
    self.assertIn('disk_format', properties)
    self.assertIn('container_format', properties)
    self.assertEqual('qcow2', properties['disk_format'])
    self.assertEqual('bare', properties['container_format'])