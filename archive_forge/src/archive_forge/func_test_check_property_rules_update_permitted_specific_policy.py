from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_check_property_rules_update_permitted_specific_policy(self):
    self.assertTrue(self.rules_checker.check_property_rules('spl_creator_policy', 'update', create_context(self.policy, ['admin'])))