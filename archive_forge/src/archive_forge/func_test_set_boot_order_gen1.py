from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_modify_virtual_system')
def test_set_boot_order_gen1(self, mock_modify_virt_syst):
    mock_vssd = self._lookup_vm()
    fake_dev_boot_order = [mock.sentinel.BOOT_DEV1, mock.sentinel.BOOT_DEV2]
    self._vmutils._set_boot_order_gen1(mock_vssd.name, fake_dev_boot_order)
    mock_modify_virt_syst.assert_called_once_with(mock_vssd)
    self.assertEqual(mock_vssd.BootOrder, tuple(fake_dev_boot_order))