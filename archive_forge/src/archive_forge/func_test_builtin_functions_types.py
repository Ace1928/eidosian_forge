import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_builtin_functions_types(self):

    class BuiltinSubclass(list):

        def bar(self, arg):
            pass
        sorted = sorted
        attr = {}
    mock = create_autospec(BuiltinSubclass)
    mock.append(3)
    mock.append.assert_called_with(3)
    self.assertRaises(AttributeError, getattr, mock.append, 'foo')
    mock.bar('foo')
    mock.bar.assert_called_with('foo')
    self.assertRaises(TypeError, mock.bar, 'foo', 'bar')
    self.assertRaises(AttributeError, getattr, mock.bar, 'foo')
    mock.sorted([1, 2])
    mock.sorted.assert_called_with([1, 2])
    self.assertRaises(AttributeError, getattr, mock.sorted, 'foo')
    mock.attr.pop(3)
    mock.attr.pop.assert_called_with(3)
    self.assertRaises(AttributeError, getattr, mock.attr, 'foo')