from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_property_protection_with_misspelt_operation(self):
    rules_with_misspelt_operation = {'^[0-9]': {'create': ['fake-role'], 'rade': ['fake-role'], 'update': ['fake-role'], 'delete': ['fake-role']}}
    self.set_property_protection_rules(rules_with_misspelt_operation)
    self.assertRaises(exception.InvalidPropertyProtectionConfiguration, property_utils.PropertyRules)