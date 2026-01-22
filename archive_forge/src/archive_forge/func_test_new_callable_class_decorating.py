import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_new_callable_class_decorating(self):
    test = self
    original = Foo

    class SomeTest(object):

        def _test(self, mock_foo):
            test.assertIsNot(Foo, original)
            test.assertIs(Foo, mock_foo)
            test.assertIsInstance(Foo, SomeClass)

        def test_two(self, mock_foo):
            self._test(mock_foo)

        def test_one(self, mock_foo):
            self._test(mock_foo)
    SomeTest = patch(foo_name, new_callable=SomeClass)(SomeTest)
    SomeTest().test_one()
    SomeTest().test_two()
    self.assertIs(Foo, original)