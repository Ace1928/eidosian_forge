import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_slots(self):

    class Foo(object):
        __slots__ = ('Foo',)
    foo = Foo()
    foo.Foo = sentinel.Foo

    @patch.object(foo, 'Foo', 'Foo')
    def anonymous():
        self.assertEqual(foo.Foo, 'Foo')
    anonymous()
    self.assertEqual(foo.Foo, sentinel.Foo)