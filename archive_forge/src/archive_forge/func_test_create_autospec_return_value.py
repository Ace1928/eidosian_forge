import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_create_autospec_return_value(self):

    def f():
        pass
    mock = create_autospec(f, return_value='foo')
    self.assertEqual(mock(), 'foo')

    class Foo(object):
        pass
    mock = create_autospec(Foo, return_value='foo')
    self.assertEqual(mock(), 'foo')