import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe, 'time')
def test_retry_if_file_in_use_exceeded_retries(self, mock_time):

    class FakeWindowsException(Exception):
        errno = errno.EACCES
    raise_count = self._handler._MAX_LOG_ROTATE_RETRIES + 1
    mock_func_side_eff = [FakeWindowsException] * raise_count
    mock_func = mock.Mock(side_effect=mock_func_side_eff)
    with mock.patch.object(namedpipe, 'WindowsError', FakeWindowsException, create=True):
        self.assertRaises(FakeWindowsException, self._handler._retry_if_file_in_use, mock_func, mock.sentinel.arg)
        mock_time.sleep.assert_has_calls([mock.call(1)] * self._handler._MAX_LOG_ROTATE_RETRIES)