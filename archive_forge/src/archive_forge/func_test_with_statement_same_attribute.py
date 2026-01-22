from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_with_statement_same_attribute(self):
    with patch('%s.something' % __name__, sentinel.Patched) as mock_something:
        self.assertEqual(something, mock_something, 'unpatched')
        with patch('%s.something' % __name__) as mock_again:
            self.assertEqual(something, mock_again, 'unpatched')
        self.assertEqual(something, mock_something, 'restored with wrong instance')
    self.assertEqual(something, sentinel.Something, 'not restored')