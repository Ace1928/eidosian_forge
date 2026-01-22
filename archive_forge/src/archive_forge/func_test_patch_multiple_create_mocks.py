import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_create_mocks(self):
    original_foo = Foo
    original_f = Foo.f
    original_g = Foo.g

    @patch.multiple(foo_name, f=DEFAULT, g=3, foo=DEFAULT)
    def test(f, foo):
        self.assertIs(Foo, original_foo)
        self.assertIs(Foo.f, f)
        self.assertEqual(Foo.g, 3)
        self.assertIs(Foo.foo, foo)
        self.assertTrue(is_instance(f, MagicMock))
        self.assertTrue(is_instance(foo, MagicMock))
    test()
    self.assertEqual(Foo.f, original_f)
    self.assertEqual(Foo.g, original_g)