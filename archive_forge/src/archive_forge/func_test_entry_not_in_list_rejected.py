from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_entry_not_in_list_rejected(self):
    check = _checks.GenericCheck('a.b.c.d', 'APPLES')
    credentials = {'a': {'b': {'c': {'d': ['PEACHES', 'PEARS']}}}}
    self.assertFalse(check({}, credentials, self.enforcer))