import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_isLinux(self) -> None:
    """
        If a system platform name is supplied to L{Platform}'s initializer, it
        is used to determine the result of L{Platform.isLinux}, which returns
        C{True} for values beginning with C{"linux"}, C{False} otherwise.
        """
    self.assertFalse(Platform(None, 'darwin').isLinux())
    self.assertTrue(Platform(None, 'linux').isLinux())
    self.assertTrue(Platform(None, 'linux2').isLinux())
    self.assertTrue(Platform(None, 'linux3').isLinux())
    self.assertFalse(Platform(None, 'win32').isLinux())