from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def testContextManagerMocking(self):
    mock = Mock()
    mock.__enter__ = Mock()
    mock.__exit__ = Mock()
    mock.__exit__.return_value = False
    with mock as m:
        self.assertEqual(m, mock.__enter__.return_value)
    mock.__enter__.assert_called_with()
    mock.__exit__.assert_called_with(None, None, None)