import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
@unittest.skipIf(six.PY3, 'No old style classes in Python 3')
def test_old_style_classes(self):

    class Foo:

        def f(self, a, b):
            pass

    class Bar(Foo):
        g = Foo()
    for spec in (Foo, Foo(), Bar, Bar()):
        mock = create_autospec(spec)
        mock.f(1, 2)
        mock.f.assert_called_once_with(1, 2)
        self.assertRaises(AttributeError, getattr, mock, 'foo')
        self.assertRaises(AttributeError, getattr, mock.f, 'foo')
    mock.g.f(1, 2)
    mock.g.f.assert_called_once_with(1, 2)
    self.assertRaises(AttributeError, getattr, mock.g, 'foo')