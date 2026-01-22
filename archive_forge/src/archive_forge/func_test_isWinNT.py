import sys
from twisted.python.reflect import namedModule
from twisted.python.runtime import Platform, shortPythonVersion
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import suppress as SUPRESS
def test_isWinNT(self) -> None:
    """
        L{Platform.isWinNT} can return only C{False} or C{True} and can not
        return C{True} if L{Platform.getType} is not C{"win32"}.
        """
    platform = Platform()
    isWinNT = platform.isWinNT()
    self.assertIn(isWinNT, (False, True))
    if platform.getType() != 'win32':
        self.assertFalse(isWinNT)