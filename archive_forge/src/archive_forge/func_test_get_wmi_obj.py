from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
@mock.patch.object(baseutils, 'wmi', create=True)
def test_get_wmi_obj(self, mock_wmi):
    result = self.utils._get_wmi_obj(mock.sentinel.moniker)
    self.assertEqual(mock_wmi.WMI.return_value, result)
    mock_wmi.WMI.assert_called_once_with(moniker=mock.sentinel.moniker)