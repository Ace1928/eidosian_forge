import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_multiple_create(self):
    patcher = patch.multiple(Foo, blam='blam')
    self.assertRaises(AttributeError, patcher.start)
    patcher = patch.multiple(Foo, blam='blam', create=True)
    patcher.start()
    try:
        self.assertEqual(Foo.blam, 'blam')
    finally:
        patcher.stop()
    self.assertFalse(hasattr(Foo, 'blam'))