from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data(True, False)
@mock.patch.object(vmutils.VMUtils, '_set_remotefx_vram')
@mock.patch.object(vmutils.VMUtils, '_get_new_resource_setting_data')
def test_set_remotefx_display_controller(self, new_obj, mock_get_new_rsd, mock_set_remotefx_vram):
    if new_obj:
        remotefx_ctrl_res = None
        expected_res = mock_get_new_rsd.return_value
    else:
        remotefx_ctrl_res = mock.MagicMock()
        expected_res = remotefx_ctrl_res
    self._vmutils._set_remotefx_display_controller(mock.sentinel.fake_vm, remotefx_ctrl_res, mock.sentinel.monitor_count, mock.sentinel.max_resolution, mock.sentinel.vram_bytes)
    self.assertEqual(mock.sentinel.monitor_count, expected_res.MaximumMonitors)
    self.assertEqual(mock.sentinel.max_resolution, expected_res.MaximumScreenResolution)
    mock_set_remotefx_vram.assert_called_once_with(expected_res, mock.sentinel.vram_bytes)
    if new_obj:
        mock_get_new_rsd.assert_called_once_with(self._vmutils._REMOTEFX_DISP_CTRL_RES_SUB_TYPE, self._vmutils._REMOTEFX_DISP_ALLOCATION_SETTING_DATA_CLASS)
        self._vmutils._jobutils.add_virt_resource.assert_called_once_with(expected_res, mock.sentinel.fake_vm)
    else:
        self.assertFalse(mock_get_new_rsd.called)
        modify_virt_res = self._vmutils._jobutils.modify_virt_resource
        modify_virt_res.assert_called_once_with(expected_res)