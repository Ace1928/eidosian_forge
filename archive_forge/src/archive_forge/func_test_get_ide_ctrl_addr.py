from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_wmi_obj')
def test_get_ide_ctrl_addr(self, mock_get_wmi_obj):
    mock_rasds = mock.Mock()
    mock_rasds.ResourceSubType = self._vmutils._IDE_CTRL_RES_SUB_TYPE
    mock_rasds.Address = mock.sentinel.ctrl_addr
    mock_get_wmi_obj.return_value = mock_rasds
    ret_val = self._vmutils._get_disk_ctrl_addr(mock.sentinel.ctrl_path)
    self.assertEqual(mock.sentinel.ctrl_addr, ret_val)
    mock_get_wmi_obj.assert_called_once_with(mock.sentinel.ctrl_path)