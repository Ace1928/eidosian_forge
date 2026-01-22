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
def test_warningLineNumberDisFindlinestarts(self):
    """
        L{deprecate.warnAboutFunction} emits a C{DeprecationWarning} with the
        number of a line within the implementation handling the case in which
        dis.findlinestarts returns the lines in random order.
        """
    from twisted_private_helper import pep626
    pep626.callTestFunction()
    warningsShown = self.flushWarnings()
    self.assertSamePath(FilePath(warningsShown[0]['filename'].encode('utf-8')), self.package.sibling(b'twisted_private_helper').child(b'pep626.py'))
    self.assertEqual(warningsShown[0]['lineno'], 15)
    self.assertEqual(warningsShown[0]['message'], 'A Warning String')
    self.assertEqual(len(warningsShown), 1)