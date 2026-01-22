from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_with_statement_specified(self):
    with patch('%s.something' % __name__, sentinel.Patched) as mock_something:
        self.assertEqual(something, mock_something, 'unpatched')
        self.assertEqual(mock_something, sentinel.Patched, 'wrong patch')
    self.assertEqual(something, sentinel.Something)