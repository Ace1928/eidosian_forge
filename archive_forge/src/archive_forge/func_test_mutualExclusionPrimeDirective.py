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
def test_mutualExclusionPrimeDirective(self):
    """
        L{mutuallyExclusiveArguments} does not interfere in its
        decoratee's operation, either its receipt of arguments or its return
        value.
        """

    @_mutuallyExclusiveArguments([('a', 'b')])
    def func(x, y, a=3, b=4):
        return x + y + a + b
    self.assertEqual(func(1, 2), 10)
    self.assertEqual(func(1, 2, 7), 14)
    self.assertEqual(func(1, 2, b=7), 13)