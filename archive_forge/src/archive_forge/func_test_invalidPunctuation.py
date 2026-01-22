from socket import AF_IPX
from twisted.internet.abstract import isIPAddress
from twisted.trial.unittest import TestCase
def test_invalidPunctuation(self) -> None:
    """
        L{isIPAddress} should return C{False} for a string containing
        strange punctuation.
        """
    self.assertFalse(isIPAddress(','))
    self.assertFalse(isIPAddress('1,2'))
    self.assertFalse(isIPAddress('1,2,3'))
    self.assertFalse(isIPAddress('1.,.3,4'))