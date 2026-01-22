from __future__ import division
import inspect
import sys
import textwrap
import six
import unittest2 as unittest
from mock import Mock, MagicMock
from mock.mock import _magics
def test_magicmock(self):
    mock = MagicMock()
    mock.__iter__.return_value = iter([1, 2, 3])
    self.assertEqual(list(mock), [1, 2, 3])
    name = '__nonzero__'
    other = '__bool__'
    if six.PY3:
        name, other = (other, name)
    getattr(mock, name).return_value = False
    self.assertFalse(hasattr(mock, other))
    self.assertFalse(bool(mock))
    for entry in _magics:
        self.assertTrue(hasattr(mock, entry))
    self.assertFalse(hasattr(mock, '__imaginery__'))