import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
def test_close_pipe(self):
    self._mock_setup_pipe_handler()
    self._handler._close_pipe()
    self._ioutils.close_handle.assert_called_once_with(mock.sentinel.pipe_handle)
    self.assertIsNone(self._handler._pipe_handle)