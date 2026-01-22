from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_property_rules_loaded_in_order(self):
    """
        Verify rules are iterable in the same order as read from the config
        file
        """
    self.rules_checker = property_utils.PropertyRules(self.policy)
    for i in range(len(property_utils.CONFIG.sections())):
        self.assertEqual(property_utils.CONFIG.sections()[i], self.rules_checker.rules[i][0].pattern)