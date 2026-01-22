import io
import json
import os
import sys
from unittest import mock
import ddt
from osprofiler.cmd import shell
from osprofiler import exc
from osprofiler.tests import test
@mock.patch('osprofiler.drivers.redis_driver.Redis.get_report')
@ddt.data(None, {'info': {'started': 0, 'finished': 1, 'name': 'total'}, 'children': []})
def test_trace_show_trace_id_not_found(self, notifications, mock_get):
    mock_get.return_value = notifications
    msg = 'Trace with UUID %s not found. Please check the HMAC key used in the command.' % self.TRACE_ID
    self._test_with_command_error(self._trace_show_cmd(), msg)