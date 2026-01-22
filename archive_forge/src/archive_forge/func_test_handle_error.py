import contextlib
from unittest import mock
from osprofiler import sqlalchemy
from osprofiler.tests import test
@mock.patch('osprofiler.sqlalchemy.profiler')
def test_handle_error(self, mock_profiler):
    original_exception = Exception('error')
    chained_exception = Exception('error and the reason')
    sqlalchemy_exception_ctx = mock.MagicMock()
    sqlalchemy_exception_ctx.original_exception = original_exception
    sqlalchemy_exception_ctx.chained_exception = chained_exception
    sqlalchemy.handle_error(sqlalchemy_exception_ctx)
    expected_info = {'etype': 'Exception', 'message': 'error', 'db': {'original_exception': str(original_exception), 'chained_exception': str(chained_exception)}}
    mock_profiler.stop.assert_called_once_with(info=expected_info)