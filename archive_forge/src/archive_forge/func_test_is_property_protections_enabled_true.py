from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_is_property_protections_enabled_true(self):
    self.config(property_protection_file='property-protections.conf')
    self.assertTrue(property_utils.is_property_protection_enabled())