import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe.NamedPipeHandler, '_close_pipe')
def test_cleanup_handles(self, mock_close_pipe):
    self._mock_setup_pipe_handler()
    log_handle = self._handler._log_file_handle
    r_event = self._handler._r_overlapped.hEvent
    w_event = self._handler._w_overlapped.hEvent
    self._handler._cleanup_handles()
    mock_close_pipe.assert_called_once_with()
    log_handle.close.assert_called_once_with()
    self._ioutils.close_handle.assert_has_calls([mock.call(r_event), mock.call(w_event)])
    self.assertIsNone(self._handler._log_file_handle)
    self.assertIsNone(self._handler._r_overlapped.hEvent)
    self.assertIsNone(self._handler._w_overlapped.hEvent)