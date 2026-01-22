from twisted.internet.error import ReactorAlreadyInstalledError
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
def test_alreadyInstalled(self) -> None:
    """
        If a reactor is already installed, L{installReactor} raises
        L{ReactorAlreadyInstalledError}.
        """
    with NoReactor():
        installReactor(object())
        self.assertRaises(ReactorAlreadyInstalledError, installReactor, object())