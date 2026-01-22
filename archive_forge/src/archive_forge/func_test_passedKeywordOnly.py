import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def test_passedKeywordOnly(self):
    """
        Keyword only arguments follow varargs.
        They are specified in PEP 3102.
        """

    def func1(*a, b=True):
        """
            b is a keyword-only argument, with a default value.
            """

    def func2(*a, b=True, c, d, e):
        """
            b, c, d, e  are keyword-only arguments.
            b has a default value.
            """
    self.assertEqual(self.checkPassed(func1, 1, 2, 3), dict(a=(1, 2, 3), b=True))
    self.assertEqual(self.checkPassed(func1, 1, 2, 3, b=False), dict(a=(1, 2, 3), b=False))
    self.assertEqual(self.checkPassed(func2, 1, 2, b=False, c=1, d=2, e=3), dict(a=(1, 2), b=False, c=1, d=2, e=3))
    self.assertRaises(TypeError, self.checkPassed, func2, 1, 2, b=False, c=1, d=2)