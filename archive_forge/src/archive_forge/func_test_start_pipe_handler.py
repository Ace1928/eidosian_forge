import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(builtins, 'open')
@mock.patch.object(namedpipe.NamedPipeHandler, '_open_pipe')
def test_start_pipe_handler(self, mock_open_pipe, mock_open):
    self._handler.start()
    mock_open_pipe.assert_called_once_with()
    mock_open.assert_called_once_with(self._FAKE_LOG_PATH, 'ab', 1)
    self.assertEqual(mock_open.return_value, self._handler._log_file_handle)
    thread = namedpipe.threading.Thread
    thread.assert_has_calls([mock.call(target=self._handler._read_from_pipe), mock.call().start(), mock.call(target=self._handler._write_to_pipe), mock.call().start()])
    for worker in self._handler._workers:
        self.assertIs(True, worker.daemon)