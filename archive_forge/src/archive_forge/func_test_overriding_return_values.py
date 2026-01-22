from warnings import catch_warnings
import unittest2 as unittest
from mock.tests.support import is_instance
from mock import MagicMock, Mock, patch, sentinel, mock_open, call
def test_overriding_return_values(self):
    mock = mock_open(read_data='foo')
    handle = mock()
    handle.read.return_value = 'bar'
    handle.readline.return_value = 'bar'
    handle.readlines.return_value = ['bar']
    self.assertEqual(handle.read(), 'bar')
    self.assertEqual(handle.readline(), 'bar')
    self.assertEqual(handle.readlines(), ['bar'])
    self.assertEqual(handle.readline(), 'bar')
    self.assertEqual(handle.readline(), 'bar')