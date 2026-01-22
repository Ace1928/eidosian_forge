from __future__ import print_function
from __future__ import absolute_import
import sys
import greenlet
from . import _test_extension
from . import TestCase
def test_instance_of_wrong_type(self):
    with self.assertRaises(TypeError) as exc:
        _test_extension.test_throw_exact(greenlet.getcurrent(), Exception(), BaseException(), None)
    self.assertEqual(str(exc.exception), 'instance exception may not have a separate value')