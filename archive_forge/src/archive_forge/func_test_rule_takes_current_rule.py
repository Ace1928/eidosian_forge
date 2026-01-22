from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_rule_takes_current_rule(self):
    results = []

    class TestCheck(object):

        def __call__(self, target, cred, enforcer, current_rule=None):
            results.append((target, cred, enforcer, current_rule))
            return False
    check = _checks.OrCheck([TestCheck()])
    self.assertFalse(check('target', 'cred', None, current_rule='a_rule'))
    self.assertEqual([('target', 'cred', None, 'a_rule')], results)