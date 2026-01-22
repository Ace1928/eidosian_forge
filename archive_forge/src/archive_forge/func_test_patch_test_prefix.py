import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
@patch('mock.patch.TEST_PREFIX', 'foo')
def test_patch_test_prefix(self):

    class Foo(object):
        thing = 'original'

        def foo_one(self):
            return self.thing

        def foo_two(self):
            return self.thing

        def test_one(self):
            return self.thing

        def test_two(self):
            return self.thing
    Foo = patch.object(Foo, 'thing', 'changed')(Foo)
    foo = Foo()
    self.assertEqual(foo.foo_one(), 'changed')
    self.assertEqual(foo.foo_two(), 'changed')
    self.assertEqual(foo.test_one(), 'original')
    self.assertEqual(foo.test_two(), 'original')