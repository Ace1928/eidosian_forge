from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_property_protection_file_doesnt_exist(self):
    self.config(property_protection_file='fake-file.conf')
    self.assertRaises(exception.InvalidPropertyProtectionConfiguration, property_utils.PropertyRules)