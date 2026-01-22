import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.datetime')
@mock.patch('osprofiler.profiler.uuidutils.generate_uuid')
@mock.patch('osprofiler.profiler.notifier.notify')
def test_profiler_start(self, mock_notify, mock_generate_uuid, mock_datetime):
    mock_generate_uuid.return_value = '44'
    now = datetime.datetime.utcnow()
    mock_datetime.datetime.utcnow.return_value = now
    info = {'some': 'info'}
    payload = {'name': 'test-start', 'base_id': '1', 'parent_id': '2', 'trace_id': '44', 'info': info, 'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%f')}
    prof = profiler._Profiler('secret', base_id='1', parent_id='2')
    prof.start('test', info=info)
    mock_notify.assert_called_once_with(payload)