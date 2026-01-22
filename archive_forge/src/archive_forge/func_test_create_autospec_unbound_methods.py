import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
@unittest.expectedFailure
def test_create_autospec_unbound_methods(self):

    class Foo(object):

        def foo(self):
            pass
    klass = create_autospec(Foo)
    instance = klass()
    self.assertRaises(TypeError, instance.foo, 1)
    klass.foo(1)
    klass.foo.assert_called_with(1)
    self.assertRaises(TypeError, klass.foo)