from socket import AF_IPX
from twisted.internet.abstract import isIPAddress
from twisted.trial.unittest import TestCase
def test_emptyString(self) -> None:
    """
        L{isIPAddress} should return C{False} for the empty string.
        """
    self.assertFalse(isIPAddress(''))