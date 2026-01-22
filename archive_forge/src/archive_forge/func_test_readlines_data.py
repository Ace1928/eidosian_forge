from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_readlines_data(self):
    mock = mock_open(read_data='foo\nbar\nbaz\n')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        result = h.readlines()
    self.assertEqual(result, ['foo\n', 'bar\n', 'baz\n'])
    mock = mock_open(read_data='foo\nbar\nbaz')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        result = h.readlines()
    self.assertEqual(result, ['foo\n', 'bar\n', 'baz'])