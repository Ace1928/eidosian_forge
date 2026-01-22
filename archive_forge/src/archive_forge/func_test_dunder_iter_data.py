import unittest
from warnings import catch_warnings
from unittest.test.testmock.support import is_instance
from unittest.mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_dunder_iter_data(self):
    mock = mock_open(read_data='foo\nbar\nbaz\n')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        lines = [l for l in h]
    self.assertEqual(lines[0], 'foo\n')
    self.assertEqual(lines[1], 'bar\n')
    self.assertEqual(lines[2], 'baz\n')
    self.assertEqual(h.readline(), '')
    with self.assertRaises(StopIteration):
        next(h)