import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple(self):
    original_foo = Foo
    original_f = Foo.f
    original_g = Foo.g
    patcher1 = patch.multiple(foo_name, f=1, g=2)
    patcher2 = patch.multiple(Foo, f=1, g=2)
    for patcher in (patcher1, patcher2):
        patcher.start()
        try:
            self.assertIs(Foo, original_foo)
            self.assertEqual(Foo.f, 1)
            self.assertEqual(Foo.g, 2)
        finally:
            patcher.stop()
        self.assertIs(Foo, original_foo)
        self.assertEqual(Foo.f, original_f)
        self.assertEqual(Foo.g, original_g)

    @patch.multiple(foo_name, f=3, g=4)
    def test():
        self.assertIs(Foo, original_foo)
        self.assertEqual(Foo.f, 3)
        self.assertEqual(Foo.g, 4)
    test()