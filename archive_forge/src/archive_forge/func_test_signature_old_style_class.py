import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
@unittest.skipIf(six.PY3, 'no old style classes in Python 3')
def test_signature_old_style_class(self):

    class Foo:

        def __init__(self, a, b=3):
            pass
    mock = create_autospec(Foo)
    self.assertRaises(TypeError, mock)
    mock(1)
    mock.assert_called_once_with(1)
    mock.assert_called_once_with(a=1)
    self.assertRaises(AssertionError, mock.assert_called_once_with, 2)
    mock(4, 5)
    mock.assert_called_with(4, 5)
    mock.assert_called_with(a=4, b=5)
    self.assertRaises(AssertionError, mock.assert_called_with, a=5, b=4)