import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
def test_open_pipe(self):
    self._handler._open_pipe()
    self._ioutils.wait_named_pipe.assert_called_once_with(mock.sentinel.pipe_name)
    self._ioutils.open.assert_called_once_with(mock.sentinel.pipe_name, desired_access=w_const.GENERIC_READ | w_const.GENERIC_WRITE, share_mode=w_const.FILE_SHARE_READ | w_const.FILE_SHARE_WRITE, creation_disposition=w_const.OPEN_EXISTING, flags_and_attributes=w_const.FILE_FLAG_OVERLAPPED)
    self.assertEqual(self._ioutils.open.return_value, self._handler._pipe_handle)