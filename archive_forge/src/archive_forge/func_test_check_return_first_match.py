from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_check_return_first_match(self):
    self.rules_checker = property_utils.PropertyRules()
    self.assertFalse(self.rules_checker.check_property_rules('x_foo_matcher', 'create', create_context(self.policy, [''])))
    self.assertFalse(self.rules_checker.check_property_rules('x_foo_matcher', 'read', create_context(self.policy, [''])))
    self.assertFalse(self.rules_checker.check_property_rules('x_foo_matcher', 'update', create_context(self.policy, [''])))
    self.assertFalse(self.rules_checker.check_property_rules('x_foo_matcher', 'delete', create_context(self.policy, [''])))