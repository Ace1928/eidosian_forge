import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.datetime')
@mock.patch('osprofiler.profiler.notifier.notify')
def test_profiler_stop(self, mock_notify, mock_datetime):
    now = datetime.datetime.utcnow()
    mock_datetime.datetime.utcnow.return_value = now
    prof = profiler._Profiler('secret', base_id='1', parent_id='2')
    prof._trace_stack.append('44')
    prof._name.append('abc')
    info = {'some': 'info'}
    prof.stop(info=info)
    payload = {'name': 'abc-stop', 'base_id': '1', 'parent_id': '2', 'trace_id': '44', 'info': info, 'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%f')}
    mock_notify.assert_called_once_with(payload)
    self.assertEqual(len(prof._name), 0)
    self.assertEqual(prof._trace_stack, collections.deque(['1', '2']))