from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_cancel_io(self):
    self._ioutils.cancel_io(mock.sentinel.handle, mock.sentinel.overlapped_struct, ignore_invalid_handle=True)
    expected_ignored_err_codes = [w_const.ERROR_NOT_FOUND, w_const.ERROR_INVALID_HANDLE]
    self._mock_run.assert_called_once_with(ioutils.kernel32.CancelIoEx, mock.sentinel.handle, self._ctypes.byref(mock.sentinel.overlapped_struct), ignored_error_codes=expected_ignored_err_codes, **self._run_args)