import copy
import re
import sys
import tempfile
from test.support import ALWAYS_EQ
import unittest
from unittest.test.testmock.support import is_instance
from unittest import mock
from unittest.mock import (
def test_magic_method_wraps_dict(self):
    data = {'foo': 'bar'}
    wrapped_dict = MagicMock(wraps=data)
    self.assertEqual(wrapped_dict.get('foo'), 'bar')
    self.assertIsInstance(wrapped_dict['foo'], MagicMock)
    self.assertFalse('foo' in wrapped_dict)
    wrapped_dict.get.return_value = 'return_value'
    self.assertEqual(wrapped_dict.get('foo'), 'return_value')
    wrapped_dict.get.return_value = sentinel.DEFAULT
    self.assertEqual(wrapped_dict.get('foo'), 'bar')
    self.assertEqual(wrapped_dict.get('baz'), None)
    self.assertIsInstance(wrapped_dict['baz'], MagicMock)
    self.assertFalse('bar' in wrapped_dict)
    data['baz'] = 'spam'
    self.assertEqual(wrapped_dict.get('baz'), 'spam')
    self.assertIsInstance(wrapped_dict['baz'], MagicMock)
    self.assertFalse('bar' in wrapped_dict)
    del data['baz']
    self.assertEqual(wrapped_dict.get('baz'), None)