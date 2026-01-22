from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_get_vm_dvd_disk_paths(self, mock_get_element_associated_class):
    self._lookup_vm()
    mock_sasd1 = mock.MagicMock(ResourceSubType=self._vmutils._DVD_DISK_RES_SUB_TYPE, HostResource=[mock.sentinel.FAKE_DVD_PATH1])
    mock_get_element_associated_class.return_value = [mock_sasd1]
    ret_val = self._vmutils.get_vm_dvd_disk_paths(self._FAKE_VM_NAME)
    self.assertEqual(mock.sentinel.FAKE_DVD_PATH1, ret_val[0])