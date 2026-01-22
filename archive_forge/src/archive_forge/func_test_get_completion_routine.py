from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import ioutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(wintypes, 'LPOVERLAPPED', create=True)
@mock.patch.object(wintypes, 'LPOVERLAPPED_COMPLETION_ROUTINE', lambda x: x, create=True)
@mock.patch.object(ioutils.IOUtils, 'set_event')
def test_get_completion_routine(self, mock_set_event, mock_LPOVERLAPPED):
    mock_callback = mock.Mock()
    compl_routine = self._ioutils.get_completion_routine(mock_callback)
    compl_routine(mock.sentinel.error_code, mock.sentinel.num_bytes, mock.sentinel.lpOverLapped)
    self._ctypes.cast.assert_called_once_with(mock.sentinel.lpOverLapped, wintypes.LPOVERLAPPED)
    mock_overlapped_struct = self._ctypes.cast.return_value.contents
    mock_set_event.assert_called_once_with(mock_overlapped_struct.hEvent)
    mock_callback.assert_called_once_with(mock.sentinel.num_bytes)