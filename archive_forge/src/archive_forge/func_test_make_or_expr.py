from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_checks, 'OrCheck', lambda x: x)
def test_make_or_expr(self):
    state = _parser.ParseState()
    result = state._make_or_expr('check1', 'or', 'check2')
    self.assertEqual([('or_expr', ['check1', 'check2'])], result)