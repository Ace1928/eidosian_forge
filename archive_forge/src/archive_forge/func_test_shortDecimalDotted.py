from socket import AF_IPX
from twisted.internet.abstract import isIPAddress
from twisted.trial.unittest import TestCase
def test_shortDecimalDotted(self) -> None:
    """
        L{isIPAddress} should return C{False} for a dotted decimal
        representation with fewer or more than four octets.
        """
    self.assertFalse(isIPAddress('0'))
    self.assertFalse(isIPAddress('0.1'))
    self.assertFalse(isIPAddress('0.1.2'))
    self.assertFalse(isIPAddress('0.1.2.3.4'))