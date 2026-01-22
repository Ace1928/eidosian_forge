from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(win32utils.Win32Utils, '_run_and_check_output')
def test_run_and_check_output_eventlet_nb_mode_disabled(self, mock_helper):
    self._win32_utils.run_and_check_output(mock.sentinel.func, mock.sentinel.arg, eventlet_nonblocking_mode=False)
    mock_helper.assert_called_once_with(mock.sentinel.func, mock.sentinel.arg)