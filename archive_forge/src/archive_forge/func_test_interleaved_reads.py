from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_interleaved_reads(self):
    mock = mock_open(read_data='foo\nbar\nbaz\n')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        line1 = h.readline()
        rest = h.readlines()
    self.assertEqual(line1, 'foo\n')
    self.assertEqual(rest, ['bar\n', 'baz\n'])
    mock = mock_open(read_data='foo\nbar\nbaz\n')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        line1 = h.readline()
        rest = h.read()
    self.assertEqual(line1, 'foo\n')
    self.assertEqual(rest, 'bar\nbaz\n')