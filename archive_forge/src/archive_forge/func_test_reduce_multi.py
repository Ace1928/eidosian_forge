from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_parser.ParseState, 'reducers', [(['tok1', 'tok2'], 'meth')])
@mock.patch.object(_parser.ParseState, 'meth', create=True, return_value=[('tok3', 'val3'), ('tok4', 'val4')])
def test_reduce_multi(self, mock_meth):
    state = _parser.ParseState()
    state.tokens = ['tok1', 'tok2']
    state.values = ['val1', 'val2']
    state.reduce()
    self.assertEqual(['tok3', 'tok4'], state.tokens)
    self.assertEqual(['val3', 'val4'], state.values)
    mock_meth.assert_called_once_with('val1', 'val2')