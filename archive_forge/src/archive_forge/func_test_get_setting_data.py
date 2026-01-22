from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_setting_data(self):
    self.netutils._get_first_item = mock.MagicMock(return_value=None)
    mock_data = mock.MagicMock()
    self.netutils._get_default_setting_data = mock.MagicMock(return_value=mock_data)
    ret_val = self.netutils._get_setting_data(self._FAKE_CLASS_NAME, self._FAKE_ELEMENT_NAME, True)
    self.assertEqual(ret_val, (mock_data, False))