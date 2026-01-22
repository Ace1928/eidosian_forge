import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe.NamedPipeHandler, '_retry_if_file_in_use')
@mock.patch.object(builtins, 'open')
@mock.patch.object(namedpipe, 'os')
def test_rotate_logs(self, mock_os, mock_open, mock_exec_retry):
    fake_archived_log_path = self._FAKE_LOG_PATH + '.1'
    mock_os.path.exists.return_value = True
    self._mock_setup_pipe_handler()
    fake_handle = self._handler._log_file_handle
    self._handler._rotate_logs()
    fake_handle.flush.assert_called_once_with()
    fake_handle.close.assert_called_once_with()
    mock_os.path.exists.assert_called_once_with(fake_archived_log_path)
    mock_exec_retry.assert_has_calls([mock.call(mock_os.remove, fake_archived_log_path), mock.call(mock_os.rename, self._FAKE_LOG_PATH, fake_archived_log_path)])
    mock_open.assert_called_once_with(self._FAKE_LOG_PATH, 'ab', 1)
    self.assertEqual(mock_open.return_value, self._handler._log_file_handle)