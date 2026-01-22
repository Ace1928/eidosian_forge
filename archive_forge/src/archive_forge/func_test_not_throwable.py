from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_not_throwable(self):
    with self.assertRaises(TypeError) as exc:
        _test_extension.test_throw_exact(greenlet.getcurrent(), 'abc', None, None)
    self.assertEqual(str(exc.exception), 'exceptions must be classes, or instances, not str')