from unittest import mock
from os_win.tests.unit import test_base
from os_win.utils.compute import rdpconsoleutils
def test_get_rdp_console_port(self):
    conn = self._rdpconsoleutils._conn
    mock_rdp_setting_data = conn.Msvm_TerminalServiceSettingData()[0]
    mock_rdp_setting_data.ListenerPort = self._FAKE_RDP_PORT
    listener_port = self._rdpconsoleutils.get_rdp_console_port()
    self.assertEqual(self._FAKE_RDP_PORT, listener_port)