from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
@mock.patch('os_win.utilsfactory.get_networkutils')
def test_netutils(self, mock_get_networkutils):
    self._hostutils._netutils_prop = None
    self.assertEqual(self._hostutils._netutils, mock_get_networkutils.return_value)