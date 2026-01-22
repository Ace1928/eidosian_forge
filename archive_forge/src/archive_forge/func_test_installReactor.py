from twisted.internet.error import ReactorAlreadyInstalledError
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
def test_installReactor(self) -> None:
    """
        L{installReactor} installs a new reactor if none is present.
        """
    with NoReactor():
        newReactor = object()
        installReactor(newReactor)
        from twisted.internet import reactor
        self.assertIs(newReactor, reactor)