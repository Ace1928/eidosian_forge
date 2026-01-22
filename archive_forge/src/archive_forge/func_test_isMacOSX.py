import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_isMacOSX(self) -> None:
    """
        If a system platform name is supplied to L{Platform}'s initializer, it
        is used to determine the result of L{Platform.isMacOSX}, which returns
        C{True} for C{"darwin"}, C{False} otherwise.
        """
    self.assertTrue(Platform(None, 'darwin').isMacOSX())
    self.assertFalse(Platform(None, 'linux2').isMacOSX())
    self.assertFalse(Platform(None, 'win32').isMacOSX())