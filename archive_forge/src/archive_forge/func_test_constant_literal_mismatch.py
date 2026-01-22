from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_constant_literal_mismatch(self):
    check = _checks.GenericCheck('True', '%(enabled)s')
    self.assertFalse(check(dict(enabled=False), {}, self.enforcer))