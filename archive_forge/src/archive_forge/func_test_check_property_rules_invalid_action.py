from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_check_property_rules_invalid_action(self):
    self.rules_checker = property_utils.PropertyRules(self.policy)
    self.assertFalse(self.rules_checker.check_property_rules('test_prop', 'hall', create_context(self.policy, ['admin'])))