import contextlib
from unittest import mock
from osprofiler import sqlalchemy
from osprofiler.tests import test
@mock.patch('osprofiler.sqlalchemy.profiler')
def test_before_execute(self, mock_profiler):
    handler = sqlalchemy._before_cursor_execute('sql')
    handler(mock.MagicMock(), 1, 2, 3, 4, 5)
    expected_info = {'db': {'statement': 2, 'params': 3}}
    mock_profiler.start.assert_called_once_with('sql', info=expected_info)