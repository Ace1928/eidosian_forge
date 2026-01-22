import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_get_async_param(self):
    """
        L{twisted.python.compat._get_async_param} uses isAsync by default,
        or deprecated async keyword argument if isAsync is None.
        """
    self.assertEqual(_get_async_param(isAsync=False), False)
    self.assertEqual(_get_async_param(isAsync=True), True)
    self.assertEqual(_get_async_param(isAsync=None, **{'async': False}), False)
    self.assertEqual(_get_async_param(isAsync=None, **{'async': True}), True)
    self.assertRaises(TypeError, _get_async_param, False, {'async': False})