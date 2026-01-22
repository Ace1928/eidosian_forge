import unittest
from warnings import catch_warnings
from unittest.test.testmock.support import is_instance
from unittest.mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_next_data(self):
    mock = mock_open(read_data='foo\nbar\nbaz\n')
    with patch('%s.open' % __name__, mock, create=True):
        h = open('bar')
        line1 = next(h)
        line2 = next(h)
        lines = [l for l in h]
    self.assertEqual(line1, 'foo\n')
    self.assertEqual(line2, 'bar\n')
    self.assertEqual(lines[0], 'baz\n')
    self.assertEqual(h.readline(), '')