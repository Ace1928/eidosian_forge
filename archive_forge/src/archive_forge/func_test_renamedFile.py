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
def test_renamedFile(self):
    """
        Even if the implementation of a deprecated function is moved around on
        the filesystem, the line number in the warning emitted by
        L{deprecate.warnAboutFunction} points to a line in the implementation of
        the deprecated function.
        """
    from twisted_private_helper import module
    del sys.modules['twisted_private_helper']
    del sys.modules[module.__name__]
    self.package.moveTo(self.package.sibling(b'twisted_renamed_helper'))
    if invalidate_caches:
        invalidate_caches()
    from twisted_renamed_helper import module
    self.addCleanup(sys.modules.pop, 'twisted_renamed_helper')
    self.addCleanup(sys.modules.pop, module.__name__)
    module.callTestFunction()
    warningsShown = self.flushWarnings([module.testFunction])
    warnedPath = FilePath(warningsShown[0]['filename'].encode('utf-8'))
    expectedPath = self.package.sibling(b'twisted_renamed_helper').child(b'module.py')
    self.assertSamePath(warnedPath, expectedPath)
    self.assertEqual(warningsShown[0]['lineno'], 9)
    self.assertEqual(warningsShown[0]['message'], 'A Warning String')
    self.assertEqual(len(warningsShown), 1)