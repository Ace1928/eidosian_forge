from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_set_vm_state(self):
    mock_vm = self._lookup_vm()
    mock_vm.RequestStateChange.return_value = (self._FAKE_JOB_PATH, self._FAKE_RET_VAL)
    self._vmutils.set_vm_state(self._FAKE_VM_NAME, constants.HYPERV_VM_STATE_ENABLED)
    mock_vm.RequestStateChange.assert_called_with(constants.HYPERV_VM_STATE_ENABLED)