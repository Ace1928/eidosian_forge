from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(wintypes, 'DWORD')
def test_write_file(self, mock_dword):
    num_bytes_written = mock_dword.return_value
    ret_val = self._ioutils.write_file(mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, mock.sentinel.overlapped_struct)
    self.assertEqual(num_bytes_written.value, ret_val)
    self._mock_run.assert_called_once_with(ioutils.kernel32.WriteFile, mock.sentinel.handle, mock.sentinel.buff, mock.sentinel.num_bytes, self._ctypes.byref(num_bytes_written), self._ctypes.byref(mock.sentinel.overlapped_struct), **self._run_args)