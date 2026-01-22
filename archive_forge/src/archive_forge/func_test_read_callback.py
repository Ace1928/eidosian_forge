import errno
from unittest import mock
from six.moves import builtins
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.io import namedpipe
from os_win.utils.winapi import constants as w_const
@mock.patch.object(namedpipe.NamedPipeHandler, '_write_to_log')
def test_read_callback(self, mock_write_to_log):
    self._mock_setup_pipe_handler()
    fake_data = self._ioutils.get_buffer_data.return_value
    self._handler._read_callback(mock.sentinel.num_bytes)
    self._ioutils.get_buffer_data.assert_called_once_with(self._handler._r_buffer, mock.sentinel.num_bytes)
    self._mock_output_queue.put.assert_called_once_with(fake_data)
    mock_write_to_log.assert_called_once_with(fake_data)