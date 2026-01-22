from unittest import mock
import ddt
from os_win.tests.unit import test_base
from os_win.utils import processutils
from os_win.utils.winapi import constants as w_const
@ddt.data({}, {'wait_exc': Exception})
@ddt.unpack
@mock.patch.object(processutils.ProcessUtils, 'open_process')
def test_wait_for_multiple_processes(self, mock_open_process, wait_exc=None):
    pids = [mock.sentinel.pid0, mock.sentinel.pid1]
    phandles = [mock.sentinel.process_handle_0, mock.sentinel.process_handle_1]
    mock_wait = self._win32_utils.wait_for_multiple_objects
    mock_wait.side_effect = wait_exc
    mock_open_process.side_effect = phandles
    if wait_exc:
        self.assertRaises(wait_exc, self._procutils.wait_for_multiple_processes, pids, mock.sentinel.wait_all, mock.sentinel.milliseconds)
    else:
        self._procutils.wait_for_multiple_processes(pids, mock.sentinel.wait_all, mock.sentinel.milliseconds)
    mock_open_process.assert_has_calls([mock.call(pid, desired_access=w_const.SYNCHRONIZE) for pid in pids])
    self._win32_utils.close_handle.assert_has_calls([mock.call(handle) for handle in phandles])
    mock_wait.assert_called_once_with(phandles, mock.sentinel.wait_all, mock.sentinel.milliseconds)