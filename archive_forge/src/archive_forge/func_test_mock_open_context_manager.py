from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_mock_open_context_manager(self):
    mock = mock_open()
    handle = mock.return_value
    with patch('%s.open' % __name__, mock, create=True):
        with open('foo') as f:
            f.read()
    expected_calls = [call('foo'), call().__enter__(), call().read(), call().__exit__(None, None, None)]
    self.assertEqual(mock.mock_calls, expected_calls)
    self.assertIs(f, handle)