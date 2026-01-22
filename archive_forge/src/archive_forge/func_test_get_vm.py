import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
def test_get_vm(self):
    expected_vm = mock.MagicMock()
    mock_conn_v2 = mock.MagicMock()
    mock_conn_v2.Msvm_ComputerSystem.return_value = [expected_vm]
    found_vm = self.liveutils._get_vm(mock_conn_v2, self._FAKE_VM_NAME)
    self.assertEqual(expected_vm, found_vm)