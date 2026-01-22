from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_supported_vm_types_2012_r2(self):
    with mock.patch.object(self._hostutils, 'check_min_windows_version') as mock_check_win:
        mock_check_win.return_value = True
        result = self._hostutils.get_supported_vm_types()
        self.assertEqual([constants.IMAGE_PROP_VM_GEN_1, constants.IMAGE_PROP_VM_GEN_2], result)