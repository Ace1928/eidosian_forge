from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_extend_or_expr(self):
    state = _parser.ParseState()
    mock_expr = mock.Mock()
    mock_expr.add_check.return_value = 'newcheck'
    result = state._extend_or_expr(mock_expr, 'or', 'check')
    self.assertEqual([('or_expr', 'newcheck')], result)
    mock_expr.add_check.assert_called_once_with('check')