from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
@mock.patch.object(baseutils, 'sys')
def test_get_wmi_conn_linux(self, mock_sys):
    mock_sys.platform = 'linux'
    result = self.utils._get_wmi_conn(mock.sentinel.moniker)
    self.assertIsNone(result)