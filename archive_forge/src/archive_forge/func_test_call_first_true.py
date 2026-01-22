from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_call_first_true(self):
    rules = [_BoolCheck(True), _BoolCheck(False)]
    check = _checks.OrCheck(rules)
    self.assertTrue(check('target', 'cred', None))
    self.assertTrue(rules[0].called)
    self.assertFalse(rules[1].called)