import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_reraiseWithNone(self):
    """
        Calling L{reraise} with an exception instance and a traceback of
        L{None} re-raises it with a new traceback.
        """
    try:
        1 / 0
    except BaseException:
        typ, value, tb = sys.exc_info()
    try:
        reraise(value, None)
    except BaseException:
        typ2, value2, tb2 = sys.exc_info()
        self.assertEqual(typ2, ZeroDivisionError)
        self.assertIs(value, value2)
        self.assertNotEqual(traceback.format_tb(tb)[-1], traceback.format_tb(tb2)[-1])
    else:
        self.fail('The exception was not raised.')