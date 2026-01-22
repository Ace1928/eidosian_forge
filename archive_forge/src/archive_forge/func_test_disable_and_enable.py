import contextlib
from unittest import mock
from osprofiler import sqlalchemy
from osprofiler.tests import test
@mock.patch('osprofiler.sqlalchemy._before_cursor_execute')
@mock.patch('osprofiler.sqlalchemy._after_cursor_execute')
def test_disable_and_enable(self, mock_after_exc, mock_before_exc):
    sqlalchemy.disable()
    sa = mock.MagicMock()
    engine = mock.MagicMock()
    sqlalchemy.add_tracing(sa, engine, 'sql')
    self.assertFalse(mock_after_exc.called)
    self.assertFalse(mock_before_exc.called)
    sqlalchemy.enable()
    sqlalchemy.add_tracing(sa, engine, 'sql')
    self.assertTrue(mock_after_exc.called)
    self.assertTrue(mock_before_exc.called)