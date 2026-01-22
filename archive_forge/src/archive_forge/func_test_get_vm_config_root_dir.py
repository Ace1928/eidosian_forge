from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_vm_config_root_dir(self):
    mock_vm = self._lookup_vm()
    config_root_dir = self._vmutils.get_vm_config_root_dir(self._FAKE_VM_NAME)
    self.assertEqual(mock_vm.ConfigurationDataRoot, config_root_dir)