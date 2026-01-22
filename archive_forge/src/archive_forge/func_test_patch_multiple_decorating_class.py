import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_decorating_class(self):
    test = self
    original_foo = Foo
    original_f = Foo.f
    original_g = Foo.g

    class SomeTest(object):

        def _test(self, f, foo):
            test.assertIs(Foo, original_foo)
            test.assertIs(Foo.f, f)
            test.assertEqual(Foo.g, 3)
            test.assertIs(Foo.foo, foo)
            test.assertTrue(is_instance(f, MagicMock))
            test.assertTrue(is_instance(foo, MagicMock))

        def test_two(self, f, foo):
            self._test(f, foo)

        def test_one(self, f, foo):
            self._test(f, foo)
    SomeTest = patch.multiple(foo_name, f=DEFAULT, g=3, foo=DEFAULT)(SomeTest)
    thing = SomeTest()
    thing.test_one()
    thing.test_two()
    self.assertEqual(Foo.f, original_f)
    self.assertEqual(Foo.g, original_g)