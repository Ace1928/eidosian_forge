from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_soft_shutdown_vm(self):
    mock_vm = self._lookup_vm()
    mock_shutdown = mock.MagicMock()
    mock_shutdown.InitiateShutdown.return_value = (self._FAKE_RET_VAL,)
    self._vmutils._conn.Msvm_ShutdownComponent.return_value = [mock_shutdown]
    self._vmutils.soft_shutdown_vm(self._FAKE_VM_NAME)
    mock_shutdown.InitiateShutdown.assert_called_once_with(Force=False, Reason=mock.ANY)
    self._vmutils._conn.Msvm_ShutdownComponent.assert_called_once_with(SystemName=mock_vm.Name)
    self._vmutils._jobutils.check_ret_val.assert_called_once_with(self._FAKE_RET_VAL, None)