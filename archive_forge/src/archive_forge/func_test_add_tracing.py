import contextlib
from unittest import mock
from osprofiler import sqlalchemy
from osprofiler.tests import test
@mock.patch('osprofiler.sqlalchemy.handle_error')
@mock.patch('osprofiler.sqlalchemy._before_cursor_execute')
@mock.patch('osprofiler.sqlalchemy._after_cursor_execute')
def test_add_tracing(self, mock_after_exc, mock_before_exc, mock_handle_error):
    sa = mock.MagicMock()
    engine = mock.MagicMock()
    mock_before_exc.return_value = 'before'
    mock_after_exc.return_value = 'after'
    sqlalchemy.add_tracing(sa, engine, 'sql')
    mock_before_exc.assert_called_once_with('sql')
    mock_after_exc.assert_called_once_with(hide_result=True)
    expected_calls = [mock.call(engine, 'before_cursor_execute', 'before'), mock.call(engine, 'after_cursor_execute', 'after'), mock.call(engine, 'handle_error', mock_handle_error)]
    self.assertEqual(sa.event.listen.call_args_list, expected_calls)