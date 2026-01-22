import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test_utils_trace_method_with_time(self):
    mock_logging = self.mock_object(utils, 'logging')
    mock_log = mock.Mock()
    mock_log.isEnabledFor = lambda x: True
    mock_logging.getLogger = mock.Mock(return_value=mock_log)
    mock_time = mock.Mock(side_effect=[3.1, 6])
    self.mock_object(time, 'time', mock_time)

    @utils.trace
    def _trace_test_method(*args, **kwargs):
        return 'OK'
    result = _trace_test_method(self)
    self.assertEqual('OK', result)
    return_log = mock_log.debug.call_args_list[1]
    self.assertIn('2900', str(return_log))