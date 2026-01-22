from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import processutils
from os_win.utils.winapi import constants as w_const
@ddt.data({}, {'assign_job_exc': Exception})
@ddt.unpack
@mock.patch.object(processutils.ProcessUtils, 'open_process')
@mock.patch.object(processutils.ProcessUtils, 'create_job_object')
@mock.patch.object(processutils.ProcessUtils, 'set_information_job_object')
@mock.patch.object(processutils.ProcessUtils, 'assign_process_to_job_object')
@mock.patch.object(processutils.kernel32_struct, 'JOBOBJECT_EXTENDED_LIMIT_INFORMATION')
def test_kill_process_on_job_close(self, mock_job_limit_struct, mock_assign_job, mock_set_job_info, mock_create_job, mock_open_process, assign_job_exc=None):
    mock_assign_job.side_effect = assign_job_exc
    mock_open_process.return_value = mock.sentinel.process_handle
    mock_create_job.return_value = mock.sentinel.job_handle
    if assign_job_exc:
        self.assertRaises(assign_job_exc, self._procutils.kill_process_on_job_close, mock.sentinel.pid)
    else:
        self._procutils.kill_process_on_job_close(mock.sentinel.pid)
    mock_open_process.assert_called_once_with(mock.sentinel.pid, w_const.PROCESS_SET_QUOTA | w_const.PROCESS_TERMINATE)
    mock_create_job.assert_called_once_with()
    mock_job_limit_struct.assert_called_once_with()
    mock_job_limit = mock_job_limit_struct.return_value
    self.assertEqual(w_const.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, mock_job_limit.BasicLimitInformation.LimitFlags)
    mock_set_job_info.assert_called_once_with(mock.sentinel.job_handle, w_const.JobObjectExtendedLimitInformation, mock_job_limit)
    mock_assign_job.assert_called_once_with(mock.sentinel.job_handle, mock.sentinel.process_handle)
    exp_closed_handles = [mock.sentinel.process_handle]
    if assign_job_exc:
        exp_closed_handles.append(mock.sentinel.job_handle)
    self._win32_utils.close_handle.assert_has_calls([mock.call(handle) for handle in exp_closed_handles])