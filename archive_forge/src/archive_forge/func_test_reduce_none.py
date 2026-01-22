from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_parser.ParseState, 'reducers', [(['tok1'], 'meth')])
@mock.patch.object(_parser.ParseState, 'meth', create=True)
def test_reduce_none(self, mock_meth):
    state = _parser.ParseState()
    state.tokens = ['tok2']
    state.values = ['val2']
    state.reduce()
    self.assertEqual(['tok2'], state.tokens)
    self.assertEqual(['val2'], state.values)
    self.assertFalse(mock_meth.called)