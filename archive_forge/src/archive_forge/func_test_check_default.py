from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_checks, 'registered_checks', {None: mock.Mock(return_value='none_check')})
def test_check_default(self):
    result = _parser._parse_check('spam:handler')
    self.assertEqual('none_check', result)
    _checks.registered_checks[None].assert_called_once_with('spam', 'handler')