import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_new_callable(self):

    class Thing(object):
        pass
    patcher = patch.multiple(Foo, f=DEFAULT, g=DEFAULT, new_callable=Thing)
    result = patcher.start()
    try:
        self.assertIs(Foo.f, result['f'])
        self.assertIs(Foo.g, result['g'])
        self.assertIsInstance(Foo.f, Thing)
        self.assertIsInstance(Foo.g, Thing)
        self.assertIsNot(Foo.f, Foo.g)
    finally:
        patcher.stop()