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
def checkOneWarning(self, modulePath):
    """
        Verification logic for L{test_deprecatedModule}.
        """
    from package import module
    self.assertEqual(FilePath(module.__file__.encode('utf-8')), modulePath)
    emitted = self.flushWarnings([self.checkOneWarning])
    self.assertEqual(len(emitted), 1)
    self.assertEqual(emitted[0]['message'], 'package.module was deprecated in Package 1.2.3: message')
    self.assertEqual(emitted[0]['category'], DeprecationWarning)