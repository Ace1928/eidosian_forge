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
def test_mutualExclusionExcludesByKeyword(self):
    """
        L{mutuallyExclusiveArguments} raises a L{TypeError}n if its
        decoratee is passed a pair of mutually exclusive arguments.
        """

    @_mutuallyExclusiveArguments([['a', 'b']])
    def func(a=3, b=4):
        return a + b
    self.assertRaises(TypeError, func, a=3, b=4)