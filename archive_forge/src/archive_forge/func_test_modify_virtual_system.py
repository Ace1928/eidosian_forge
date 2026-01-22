from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_modify_virtual_system(self):
    mock_vs_man_svc = self._vmutils._vs_man_svc
    mock_vmsetting = mock.MagicMock()
    fake_job_path = 'fake job path'
    fake_ret_val = 'fake return value'
    mock_vs_man_svc.ModifySystemSettings.return_value = (fake_job_path, fake_ret_val)
    self._vmutils._modify_virtual_system(vmsetting=mock_vmsetting)
    mock_vs_man_svc.ModifySystemSettings.assert_called_once_with(SystemSettings=mock_vmsetting.GetText_(1))
    self._vmutils._jobutils.check_ret_val.assert_called_once_with(fake_ret_val, fake_job_path)