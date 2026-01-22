import os
import sys
import six
import unittest2 as unittest
from mock.tests import support
from mock.tests.support import SomeClass, is_instance, callable
from mock import (
from mock.mock import _patch, _get_target
def test_patch_imports_lazily(self):
    sys.modules.pop('squizz', None)
    p1 = patch('squizz.squozz')
    self.assertRaises(ImportError, p1.start)
    squizz = Mock()
    squizz.squozz = 6
    sys.modules['squizz'] = squizz
    p1 = patch('squizz.squozz')
    squizz.squozz = 3
    p1.start()
    p1.stop()
    self.assertEqual(squizz.squozz, 3)