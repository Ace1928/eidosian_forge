import re
from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils10
@mock.patch.object(hostutils10.HostUtils10, '_get_wmi_conn')
def test_conn_hgs_no_namespace(self, mock_get_wmi_conn):
    self._hostutils._conn_hgs_attr = None
    mock_get_wmi_conn.side_effect = [exceptions.OSWinException]
    self.assertRaises(exceptions.OSWinException, lambda: self._hostutils._conn_hgs)
    mock_get_wmi_conn.assert_called_once_with(self._hostutils._HGS_NAMESPACE % self._hostutils._host)