import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test_utils_trace_method_with_exception(self):
    self.LOG = self.mock_object(utils, 'LOG')

    @utils.trace
    def _trace_test_method(*args, **kwargs):
        raise exception.VolumeDeviceNotFound('test message')
    self.assertRaises(exception.VolumeDeviceNotFound, _trace_test_method)
    exception_log = self.LOG.debug.call_args_list[1]
    self.assertIn('exception', str(exception_log))
    self.assertIn('test message', str(exception_log))